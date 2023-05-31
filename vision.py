import torchvision.utils as vutils
from model import *
# 可视化cutmix、cutout、mixup的效果

# 取三张训练样本
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images[:3]
labels = labels[:3]
print(images)
print(labels)

raw_images = images.clone()

raw_images = vutils.make_grid(raw_images, normalize=False, scale_each=True)
raw_images = raw_images.cpu().numpy().transpose(1, 2, 0)
raw_images = np.clip(raw_images, 0, 1)
plt.imshow(raw_images)
plt.title('Raw')
plt.show()


# cutmix
mixed_images = images.clone()
lam = np.random.beta(1, 1)
rand_index = torch.randperm(images.size()[0])
bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
mixed_labels_a = labels
mixed_labels_b = labels[rand_index]

mixed_images = vutils.make_grid(mixed_images, normalize=True, scale_each=True)
mixed_images = mixed_images.cpu().numpy().transpose(1, 2, 0)
plt.imshow(mixed_images)
plt.title('CutMix')
plt.show()


#cutout
cutout_images = images.clone()
cutout_images = cutout(cutout_images, n_holes=1, length=16)

cutout_images = vutils.make_grid(cutout_images, normalize=True, scale_each=True)
cutout_images = cutout_images.cpu().numpy().transpose(1, 2, 0)
plt.imshow(cutout_images)
plt.title('Cutout')
plt.show()


# mixup
mixed_images = mixup_data(images, labels, alpha=1.0)[0]

mixed_images = vutils.make_grid(mixed_images, normalize=True, scale_each=True)
mixed_images = mixed_images.cpu().numpy().transpose(1, 2, 0)
plt.imshow(mixed_images)
plt.title('Mixup')
plt.show()
