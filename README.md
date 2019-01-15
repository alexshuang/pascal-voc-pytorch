# 搞定目标检测（SSD篇）

![](https://upload-images.jianshu.io/upload_images/13575947-08e4cd04dd185415.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

通过对[搞定目标检测（SSD篇）（上）](https://www.jianshu.com/p/8d894605bb06)的学习，你应该已经了解目标检测的基本原理和技术局限性，本文将会详解如何实现SSD目标检测模型。先打个预防针，本文的内容会比较烧脑，而且默认你已经掌握了[上集](https://www.jianshu.com/p/8d894605bb06)的内容，当然我也会用平实的语言尽力给你讲清楚。Github: [https://github.com/alexshuang/pascal-voc-pytorch](https://github.com/alexshuang/pascal-voc-pytorch)。

## SSD / [Paper](https://arxiv.org/abs/1512.02325) / [Notebook](https://github.com/alexshuang/pascal-voc-pytorch/blob/master/pascal_voc2012_ssd.ipynb)

从SSD的全称，Single Shot MultiBox Detector，就可以窥探算法的本质：“Single Shot”指的是单目标检测，“MultiBox”中的“Box”就像是我们平时拍摄时用到的取景框，只关注框内的画面，屏蔽框外的内容。创建“Multi”个"Box"，将每个"Box"的单目标检测结果汇总起来就是多目标检测。换句话说，SSD将图像切分为N片，并对每片进行独立的单目标检测，最后汇总每片的检测结果。

![Figure 1: SSD arch](https://upload-images.jianshu.io/upload_images/13575947-47dbeb80969ad6ac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

SSD切分图像的方法就是Convolution，或者说是Receptive Field。如架构图所示，SSD的top layers（extra feature layers）是卷积层。假设卷积计算的结果是：[64, 25, 4, 4]，它指的是在4x4大小的feature map中，每个grid cell代表了原始图像中的一个区域。换句话说，如果用4x4的网格平铺整个图像，feature map中的每个grid cell对应一个网格区域。

![Figure 2: 4x4 grid cells](https://upload-images.jianshu.io/upload_images/13575947-4eebfe8d0f62cc91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Figure 2中的网格就是MultiBox，每个Box的单目标检测的结果就保存在卷积运算的channels维度。对于矩阵 [64, 25, 4, 4]，25 = bounding box + 分类概率（额外增加"Background"分类） = 4 + 21。

如Figure 1所示，SSD通过pooling层（or stride）不断调整网格的数量，例如从4x4 -> 2x2 -> 1x1，并将所有结果汇总起来，这样就可以使用不同大小的Box来锚定不同大小的物体。

## Classification

延续上集的思路，我将多目标检测也分解为分类（Classification）和定位（Location）两个独立操作。相比单目标检测，多目标检测模型最终用sigmoid()而不是softmax()来生成分类的概率。为检验Classification模型的预测准确率，我选取所有概率大于thresh（0.4）的分类，可以看到，Classification模型是work的（降低thresh可以显示更多的分类）。

![](https://upload-images.jianshu.io/upload_images/13575947-ec8c0b3fa708ba48.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Ground Truth （Location）

Ground Truth指的是图像的标注信息，如classification、bounding box、segmentation等信息。

```
i = 9
bb = y[0][i].view(-1, 4)
clas = y[1][i]
bb, clas, bb.shape, clas.shape

(tensor([[  0.,   0.,   0.,   0.],
         [  0.,   0.,   0.,   0.],
         [  0.,   0.,   0.,   0.],
         [105.,   4., 161.,  28.],
         [ 70.,   0., 149.,  66.],
         [ 50.,  24., 185., 129.],
         [ 19.,  60., 223., 222.]], device='cuda:0'),
 tensor([ 0,  0,  0,  4, 14, 14, 14], device='cuda:0'),
 torch.Size([7, 4]),
 torch.Size([7]))
```

由于每个训练样本的ground truth个数不同，为了保证mini-batch矩阵的一致性，Pytorch会用0来填充y矩阵，因此，在使用数据时，需要先剔除bounding box全0的ground truth。

```
i = 9
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(ima[i])
draw_gt(ax, y[0][i].view(-1, 4), y[1][i], num_classes=len(labels))
ax.axis('off')
```

![](https://upload-images.jianshu.io/upload_images/13575947-6cb5940151cbb60e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

## SSD Network Part 1

```
def conv_layer(nin, nf, stride=2, drop=0.1):
  return nn.Sequential(
      nn.Conv2d(nin, nf, 3, stride, 1, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(nf),
      nn.Dropout(drop)
  )

class Outlayer(nn.Module):
  def __init__(self, nf, num_classes, bias):
    super().__init__()
    self.clas_conv = nn.Conv2d(nf, num_classes + 1, 3, 1, 1)
    self.bb_conv = nn.Conv2d(nf, 4, 3, 1, 1)
    self.clas_conv.bias.data.zero_().add_(bias)
    
  def flatten(self, x):
    bs, nf, w, h = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf)
  
  def forward(self, x):
    return [self.flatten(self.bb_conv(x)), self.flatten(self.clas_conv(x))]

class SSDHead(nn.Module):
  def __init__(self, num_classes, nf, bias, drop_i=0.25):
    super().__init__()
    self.conv1 = conv_layer(512, nf, stride=1)
    self.conv2 = conv_layer(nf, nf)
    self.drop_i = nn.Dropout(drop_i)
    self.out = Outlayer(nf, num_classes, bias=bias)
  
  def forward(self, x):
    x = self.drop_i(F.relu(x))
    x = self.conv1(x)
    x = self.conv2(x)
    return self.out(x)
  
ssd_head_f = SSDHead(num_classes, nf, bias=-3.)
```

我使用的backbone是Resnet34，它最终的输出结果是7x7x512，经过stride=2的conv2()后，得到如Figure 2所示的4x4 freature map。通过Outlayer()生成channels分别为4和21的输出，前者与bounding box相关，后者是分类概率。之所以将clas_conv层的bias初始化为-3，是因为模型输出的总loss值偏大，虽然可以通过后续训练降低loss值，但模型却达不到期望效果，通过初始化赋值可以解决这个问题。

> 为什么是“与bounding box有关”，而不是bounding box？
[搞定目标检测（SSD篇）（上）](https://www.jianshu.com/p/8d894605bb06)已经提到，Resnet这类图像识别模型并不擅长生成空间数据，因此SSD生成的并不是bounding box，而是bounding box相对于default box的偏移（offset），而default box则是预先定义好的bounding box，如Figure 2中的网格。

## Default Box

Default Box就是“MultiBox”，是SSD的取景框，即Figure 2中的网格。它由[中心x、y坐标，width，height]组成。

```
cells = 4
width = 1 / cells
cx = np.repeat(np.linspace(width / 2, 1 - (width / 2), cells), cells)
cy = np.tile(np.linspace(width / 2, 1 - (width / 2), cells), cells)
w = h = np.array([width] * cells**2)
def_box = T(np.stack([cx, cy, w, h], 1))
def_box

tensor([[0.1250, 0.1250, 0.2500, 0.2500],
        [0.1250, 0.3750, 0.2500, 0.2500],
        [0.1250, 0.6250, 0.2500, 0.2500],
        [0.1250, 0.8750, 0.2500, 0.2500],
        [0.3750, 0.1250, 0.2500, 0.2500],
        [0.3750, 0.3750, 0.2500, 0.2500],
        [0.3750, 0.6250, 0.2500, 0.2500],
        [0.3750, 0.8750, 0.2500, 0.2500],
        [0.6250, 0.1250, 0.2500, 0.2500],
        [0.6250, 0.3750, 0.2500, 0.2500],
        [0.6250, 0.6250, 0.2500, 0.2500],
        [0.6250, 0.8750, 0.2500, 0.2500],
        [0.8750, 0.1250, 0.2500, 0.2500],
        [0.8750, 0.3750, 0.2500, 0.2500],
        [0.8750, 0.6250, 0.2500, 0.2500],
        [0.8750, 0.8750, 0.2500, 0.2500]], device='cuda:0')
```

> 你是否注意到Figure 2图中的很多网格被识别为background，是模型错了？
Figure 2是default boxes和ground truth相互匹配后得到的结果，实际上，因为没有和ground truth大小相似的default box，因此只能选择最适配的default box，但因为两者大小相差悬殊，所以才产生了错配的感觉。

## Jaccard Index

default box和ground truth是通过jaccard index相互匹配的。通过jaccard()计算每个default box和每个ground truth的交并比 -- overlap，那些overlap > 0.5的default box index，就是jaccard index。通过jaccard index，可以知道default box对应哪个ground truth。

![](https://upload-images.jianshu.io/upload_images/13575947-73fcb1fd4e5659c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
def box_size(box): return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

def intersection(gt, def_box):
  left_top = torch.max(gt[:, None, :2], def_box[None, :, :2])
  right_bottom = torch.min(gt[:, None, 2:], def_box[None, :, 2:])
  wh = torch.clamp(right_bottom - left_top, min=0)
  return wh[:, :, 0] * wh[:, :, 1]
  
def jaccard(gt, def_box):
  inter = intersection(gt, def_box)
  union = box_size(gt).unsqueeze(1) + box_size(def_box).unsqueeze(0) - inter
  return inter / union

overlap = jaccard(bb, def_box_bb * sz)
gt_best_overlap, gt_db_idx = overlap.max(1)
db_best_overlap, db_gt_idx = overlap.max(0)
db_best_overlap[gt_db_idx] = 1.1
is_obj = db_best_overlap > 0.5
pos_idxs = np.nonzero(is_obj)[:, 0]
neg_idxs = np.nonzero(1 - is_obj)[:, 0]
db_clas = T([num_classes] * len(db_best_overlap))
db_clas[pos_idxs] = clas[db_gt_idx[pos_idxs]]
db_best_overlap, db_clas
```

db_gt_idx指的是，每个default box对应的ground truth id。db_best_overlap是指每个default box内最大的jaccard，jaccard最大的default_box也不一定都满足> 0.5的要求（如Figure 2），所以主动将ground truth所对应的default_box的overlap提升为1.1。db_clas就是jaccard index。

## More Default Boxes

还记得SSD的架构吗，随着extra feature layers的深入，feature map的网格越来越大，从4x4->2x2->1x1，也就是说，它可以匹配更多体型的物体。除此之外，SSD还会利用不同的宽纵比，创建大小相同但形状不同的default box：

![](https://upload-images.jianshu.io/upload_images/13575947-2cac943e12e99d62.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如图所示，每个default box可以分为3大类：宽比高长、高比宽长、等长，所以我采用的宽纵比：[(1., 1.), (1., 0.5), (0.5, 1.)]，并为每类都配置了scale系数：[0.7, 1., 1.3]。

```
cells = np.array([4, 2, 1])
center_offsets = 1 / cells / 2
aspect_ratios = [(1., 1.), (1., .5), (.5, 1.)]
zooms = [0.7, 1., 1.3]
scales = [(o * i, o * j) for o in zooms for i, j in aspect_ratios]
k = len(scales)
k, scales

(9,
 [(0.7, 0.7),
  (0.7, 0.35),
  (0.35, 0.7),
  (1.0, 1.0),
  (1.0, 0.5),
  (0.5, 1.0),
  (1.3, 1.3),
  (1.3, 0.65),
  (0.65, 1.3)])
```

k就是每个default box根据宽纵比产生的变化数。如果把default box比作相机，k则是为该相机配备的专业镜头数，不同拍摄场景使用不同的镜头。

![Figure 3: (4x4 + 2x2 + 1x1) * k grid cells](https://upload-images.jianshu.io/upload_images/13575947-425120d2063c003c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可以看到，Figure 3就比Figure 2精确很多，当然它的default box也比后者多得多： (4x4 + 2x2 + 1x1) * k。

## Loss Function

SSD的损失函数和我们在[上集](https://www.jianshu.com/p/8d894605bb06)的类似，分别计算bounding box loss（loc loss）和classification loss（conf loss），它们的总和就是最终loss:

![loss function.png](https://upload-images.jianshu.io/upload_images/13575947-629062249c081da0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

loc loss是经bounding box offset（SSD模型的输出）修正后的default box和ground truth的L1 loss。conf loss则是binary cross entropy。

```
class BCELoss(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.num_classes = num_classes
    
  def get_weight(self, x, t): return None
  
  def forward(self, x, t):
    x = x[:, :-1]
    one_hot_t = torch.eye(num_classes + 1)[t.data.cpu()]
    t = V(one_hot_t[:, :-1].contiguous())
    w = self.get_weight(x, t)
    return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / self.num_classes

bce_loss_f = BCELoss(num_classes)

def loc_loss(preds, targs):
  return (preds - targs).abs().mean()

def conf_loss(preds, targs):
  return bce_loss_f(preds, targs)
```

BCELoss去掉background分类的预测结果是因为_ssd_loss()构建的db_clas包含了不属于数据集的background分类。之所以要将conf_loss的结果除以self.num_classes，是因为如果binary cross entropy采用sum而不非mean来处理loss，conf_loss就会偏大，反之如果采用mean来处理，conf_loss就会偏小，不管是偏大还是偏小，都不利于模型训练，所以解决方法就是采用像前文的bias初始化那样主动降低loss值，这里采用的方法是除以20。

```
def offset_to_bb(off, db_bb):
    off = F.tanh(off)
    center = (off[:, :2] / 2) * db_bb[:, 2:] + db_bb[:, :2]
    wh = ((off[:, 2:] / 2) + 1) * db_bb[:, 2:]
    return def_box_to_bb(center, wh)

def _ssd_loss(db_offset, clas, bb_gt, clas_gt):
  bb = offset_to_bb(db_offset, def_box)
  bb_gt = bb_gt.view(-1, 4) / sz
  idxs = np.nonzero(bb_gt[:, 2] > 0)[:, 0]
  bb_gt, clas_gt = bb_gt[idxs], clas_gt[idxs]
  overlap = jaccard(bb_gt, def_box_bb)
  gt_best_overlap, gt_db_idx = overlap.max(1)
  db_best_overlap, db_gt_idx = overlap.max(0)
  db_best_overlap[gt_db_idx] = 1.1
  for i, o in enumerate(gt_db_idx): db_gt_idx[o] = i
  is_obj = db_best_overlap >= 0.5
  pos_idxs = np.nonzero(is_obj)[:, 0]
  neg_idxs = np.nonzero(1 - is_obj.data)[:, 0]
  db_clas = clas_gt[db_gt_idx]
  db_clas[neg_idxs] = len(labels)
  db_bb = bb_gt[db_gt_idx]
  return (loc_loss(bb[pos_idxs], db_bb[pos_idxs]), bce_loss_f(clas, db_clas))

def ssd_loss(preds, targs, print_loss=False):
#   alpha = 1.
  loc_loss, conf_loss = 0., 0.
  for i, (db_offset, clas, bb_gt, clas_gt) in enumerate(zip(*preds, *targs)):
    losses = _ssd_loss(db_offset, clas, bb_gt, clas_gt)
    loc_loss += losses[0]# * alpha
    conf_loss += losses[1]
  if print_loss:
    print(f'loc loss: {loc_loss:.2f}, conf loss: {conf_loss:.2f}')
  return loc_loss + conf_loss
```

_ssd_loss()中，offset_to_bb()的作用就是根据bounding box offset来修正default box。bounding box offset的值是default box的scale系数，不仅移动default box的位置，还会改变default box的宽高。_ssd_loss()中很多代码在前面已经讲解过了，其目的就是根据ground truth重新构建以default box为基础的ground truth，之所以这样做是因为我们要预测每个default box中的分类。

## Train 4x4

终于来到模型训练阶段了，为了便于调试，我们先只训练4x4网格模型，使用"SSD Network Part 1"定义的模型。

```
lr = 1e-2
learn.fit(lr, 1, cycle_len=8, use_clr=(20, 5))
learn.save('16')

epoch      trn_loss   val_loss   
    0      33.574218  34.117771 
    1      30.093091  29.408577 
    2      27.206728  27.568285 
    3      25.348878  26.957813 
    4      23.976828  26.765239 
    5      22.80882   26.695604 
    6      21.532631  26.688388 
    7      20.018111  26.610572 
```
![](https://upload-images.jianshu.io/upload_images/13575947-f47c49ab24ef7447.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从测试结果可以看到，default box不再像之前那样整齐划一，box的大小也略有不同。它们看似凌乱，但实际上都是基于原来的位置的偏移，这点从box编号可以看出。总体来说，模型预测结果比静态default box要更准确些。接下来我们来增加更多的default box。

## SSD Network Part 2

```
class Outlayer(nn.Module):
  def __init__(self, nf, num_classes, bias):
    super().__init__()
    self.clas_conv = nn.Conv2d(nf, (num_classes + 1) * k, 3, 1, 1)
    self.bb_conv = nn.Conv2d(nf, 4 * k, 3, 1, 1)
    self.clas_conv.bias.data.zero_().add_(bias)
  
  def flatten(self, x):
    bs, nf, w, h = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf // k)
  
  def forward(self, x):
    return [self.flatten(self.bb_conv(x)), self.flatten(self.clas_conv(x))]

class SSDHead(nn.Module):
  def __init__(self, num_classes, nf, bias, drop_i=0.25, drop_h=0.1):
    super().__init__()
    self.conv1 = conv_layer(512, nf, stride=1, drop=drop_h)
    self.conv2 = conv_layer(nf, nf, drop=drop_h)   # 4x4
    self.conv3 = conv_layer(nf, nf, drop=drop_h)   # 2x2
    self.conv4 = conv_layer(nf, nf, drop=drop_h)   # 1x1
    self.drop_i = nn.Dropout(drop_i)
    self.out1 = Outlayer(nf, num_classes, bias)
    self.out2 = Outlayer(nf, num_classes, bias)
    self.out3 = Outlayer(nf, num_classes, bias)
  
  def forward(self, x):
    x = self.drop_i(F.relu(x))
    x = self.conv1(x)
    x = self.conv2(x)
    bb1, clas1 = self.out1(x)
    x = self.conv3(x)
    bb2, clas2 = self.out2(x)
    x = self.conv4(x)
    bb3, clas3 = self.out3(x)
    return [torch.cat([bb1, bb2, bb3], 1),
            torch.cat([clas1, clas2, clas3], 1)]

drops = [0.4, 0.2]
ssd_head_f = SSDHead(num_classes, nf, -4., drop_i=drops[0], drop_h=drops[1])
```
SSD将4x4、2x2、1x1三种不同大小的detector的预测结果汇总在一起，因为每个default box会有k种变化，所以每个detector的输出是原来的k倍。从之前的训练结果来看，现在正则化程度不够，所以我增加dropout的概率。

```
lr = 1e-2
learn.fit(lr, 1, cycle_len=10, use_clr=(20, 10))
learn.save('multi')

epoch      trn_loss   val_loss   
    0      87.026507  75.858966 
    1      68.657919  62.675859 
    2      58.815842  78.257847 
    3      53.675965  54.85459  
    4      49.656684  53.707109 
    5      46.777794  53.003534 
    6      44.20865   51.358076 
    7      41.394307  51.515281 
    8      38.741202  50.559135 
    9      36.69472   50.12559  
```
![](https://upload-images.jianshu.io/upload_images/13575947-e6caf5a5f83f01e5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看到，bounding box比原来的要大，这正是我们所希望看到的，但桌子上的酒瓶却没有被框定，原因何在？原因在于调整后的default box整体比之前偏大，因为酒瓶比较小，所以它的overlap < 0.5，无法被定位，所以最有效的解决方法是减少overlap thresh，比如将overlap thresh调整为0.4。

## NMS

SSD模型的最后一层是nms，它的作用就是对筛选出那些大于某个jaccard overlap thresh的bounding box，我选出的是jaccard overlap > 0.4的前50个bounding box用于测试。

![](https://upload-images.jianshu.io/upload_images/13575947-8fa43060bac29343.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从结果上看，并不是我们想要的结果，4个目标只有1个被检测出来了， 这又是为何？

```
x, y = next(iter(md.trn_dl))
yp = predict_batch(learn.model, x)
ssd_loss(yp, y, True)

loc loss: 3.65, conf loss: 28.08
tensor(31.7384, device='cuda:0', grad_fn=<AddBackward0>)
```

原因就在于conf_loss太大，classification准确率低。从神经网络模型来看，location和classification只有最后一层是独立，其他层都是共享的，换句话说，如果classification准确率低，那location的准确率也高不到拿去，实际上，location是依赖于classification的，先识别再定位。

## Focal Loss / [Paper](https://arxiv.org/abs/1708.02002)

![](https://upload-images.jianshu.io/upload_images/13575947-ff420a35c83e81e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从数学公式可以看出，focal loss是scale版的cross entropy，$-(1 - p_t)^\gamma$是可训练的scale值。在object dection中，focal loss的表现远胜于BCE，其背后的逻辑是：通过scale（放大/缩小）输出，将原本模糊不清的预测确定化。当gamma == 0时，focal loss就相当于corss entropy(CE)，如蓝色曲线所示，即使probability达到0.6，loss值还是>= 0.5，就好像是说：“我判断它不是分类B的概率是60%，恩，我还有继续努力优化参数，我行的”。当gamma == 2时，同样是probability达到0.6，loss值接近于0，就好像是说：“我判断它不是分类B的概率是60%，恩，根据我多年断案经验，它一定不是分类B，好了，虽然预测准确性不算高，但没关系，结案了，接下来我们应该把精力投入到那些准确率还很低的项目中，加油吧”。

focal loss会对well-classified examples降权，即降低它的loss值，也就是减少参数更新值，把更多优化空间留给预测概率较低的样本，从整体角度来优化模型。

```
class FocalLoss(BCELoss):
  def get_weight(self, x, t):
    alpha,gamma = 0.25,1
    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)
    w = alpha*t + (1-alpha)*(1-t)
    return w * (1-pt).pow(gamma)

bce_loss_f = FocalLoss(num_classes)
lr = 1e-2
learn.fit(lr, 1, cycle_len=10, use_clr=(20, 10))
learn.save('focal_loss')

epoch      trn_loss   val_loss   
    0      17.30767   18.866698 
    1      15.211579  13.772004 
    2      13.563804  13.015255 
    3      12.589626  12.785115 
    4      11.926406  12.28807  
    5      11.515744  11.814605 
    6      11.109133  11.686357 
    7      10.664063  11.424233 
    8      10.285392  11.338397 
    9      9.935587   11.185435 
```

![](https://upload-images.jianshu.io/upload_images/13575947-5b6dfc604ba62249.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和预期一样，虽然主体物体detector的预测准确率降低了（从0.77降低到0.5），但其他物体detector的预测准确率也提升了。除了酒瓶之外（原因前面已经分析了），另外三个物体都被准确检测出来。

## END

SSD就像一个没有天赋但却很勤奋的摄影师，每次拍摄他都遵循同一套流程，取景、移动镜头到取景框中心位置、咔嚓一声摁下快门，但他又是了不起的，可以不厌其烦地选取各个拍摄角度和各种取景框。到这里，我已经完成了对SSD算法理解的分享，这趟旅程可能会比较烧脑，你需要结合[代码]()和paper来学习。
