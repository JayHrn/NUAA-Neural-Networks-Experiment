import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split#train_test_split：数据集划分。
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc#classification_report 和 roc_curve：分类报告和评估。
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.datasets import fetch_lfw_people
from PIL import Image#PIL.Image 用于处理图像（灰度化、缩放）
from time import time

# 数据集加载类
"""实现一个数据集加载类 my_dataset，用于加载和处理两种人脸数据集：LFW (Labeled Faces in the Wild) 和 
ORL (Olivetti Research Laboratory Faces)，并将其划分为训练集和测试集"""
class my_dataset:
    def __init__(self, dataset_type="lfw", data_path=None, min_faces_per_person=70, resize=0.4):
        """
        初始化数据集加载类，支持 LFW 和 ORL 数据集。
        :param dataset_type: 数据集类型 ("lfw" 或 "orl")。
        :param data_path: 数据集路径，ORL 数据集需要指定。
        :param min_faces_per_person: 最小人脸数量过滤 (仅适用于 LFW)。
        :param resize: 图像缩放比例。
        """
        self.dataset_type = dataset_type
        if dataset_type == "lfw":
            self.dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=resize, data_home=data_path)
            self.images = self.dataset.images
            self.data = self.dataset.data
            self.target = self.dataset.target
            self.target_names = self.dataset.target_names
        elif dataset_type == "orl":
            self.images, self.data, self.target, self.target_names = self.load_orl_dataset(data_path, resize)
        else:
            raise ValueError("Unsupported dataset type. Use 'lfw' or 'orl'.")

        self.n_samples, self.h, self.w = self.images.shape
        self.n_features = self.data.shape[1]
        self.n_classes = len(self.target_names)
        # 加载完成后提取以下信息：
        # self.n_samples: 样本数，即图片总数量。
        # self.h, self.w: 每张图片的高度和宽度（像素）。
        # self.n_features: 每张图片展平后的特征数量。
        # self.n_classes: 类别数（不同的人数）

        print("样本数: %d" % self.n_samples)
        print("选择的人物个数: %d" % self.n_classes)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.25, random_state=42)

    """功能是加载 ORL (Olivetti Research Laboratory Faces) 数据集并对其进行预处理，包括灰度化、缩放、展平和生成目标标签，以便后续用于机器学习任务"""
    def load_orl_dataset(self, data_path, resize):
        """
         方法的输入参数
data_path: ORL 数据集的路径。假定数据集按照标准格式存储，即每个人的图片存储在单独的子目录中，每个子目录名为对应人物的标识。
resize: 图像缩放比例，用于调整图像尺寸，减小数据规模以节省计算资源。
2. 方法的返回值
images: 一个三维数组，包含所有图像的像素值，维度为 [样本数, 缩放后高度, 缩放后宽度]。
data: 一个二维数组，每张图像展平成一维向量，维度为 [样本数, 特征数]。
target: 一个一维数组，存储每张图像对应的类别标签。
target_names: 一个列表，存储类别名称（即子目录的名称，通常是人物的标识
        """
        images = []
        target = []
        target_names = []
        for person_id, person_dir in enumerate(sorted(os.listdir(data_path))):
            person_path = os.path.join(data_path, person_dir)
            if not os.path.isdir(person_path):
                continue
            for img_name in sorted(os.listdir(person_path)):
                img_path = os.path.join(person_path, img_name)
                img = Image.open(img_path).convert('L')  # 灰度图
                img_resized = img.resize((int(img.width * resize), int(img.height * resize)))
                images.append(np.array(img_resized))
                target.append(person_id)
            target_names.append(person_dir)
        images = np.array(images)
        data = images.reshape(images.shape[0], -1)  # 展平成一维
        target = np.array(target)
        return images, data, target, target_names

    """功能是返回已经划分好的训练集和测试集数据。"""
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

# 降维类
"""定义了一个类 my_decomposition 的初始化方法 __init__，用于对数据进行降维处理的准备工作。"""
class my_decomposition:
    def __init__(self, dataset, choice, n_components):
        self.dataset = dataset
        self.choice = choice
        self.X_train, self.X_test, self.y_train, self.y_test = self.dataset.get_data()
        self.decom(choice, n_components)
    """定义了 decom 方法，负责对数据集进行降维处理，支持多种降维算法。降维的目的是将高维数据映射到较低维空间，
    降低数据维度，同时尽量保留原始数据的结构和信息。
    支持的降维算法：
PCA (主成分分析)：线性降维算法，保留最大方差方向。
KPCA (核主成分分析)：非线性降维，通过核函数映射数据到高维后再降维。
LLE (局部线性嵌入)：保持局部邻域关系的非线性降维算法。
Isomap：保持全局流形结构的降维算法。
MDS (多维缩放)：保持样本之间距离关系的降维方法。"""
    def decom(self, choice, n_components):
        t0 = time()
        if self.choice == "pca+svm":
            self.re_model = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(self.X_train)
            self.X_train_re = self.re_model.transform(self.X_train)
            self.X_test_re = self.re_model.transform(self.X_test)
        elif self.choice == "kpca+svm":
            self.re_model = KernelPCA(n_components=n_components, kernel='cosine').fit(self.X_train)
            self.X_train_re = self.re_model.transform(self.X_train)
            self.X_test_re = self.re_model.transform(self.X_test)
        elif self.choice == "lle+svm":
            self.re_model = LocallyLinearEmbedding(n_components=n_components, n_neighbors=200).fit(self.X_train)
            self.X_train_re = self.re_model.transform(self.X_train)
            self.X_test_re = self.re_model.transform(self.X_test)
        elif self.choice == "isomap+svm":
            self.re_model = Isomap(n_components=n_components, n_neighbors=200).fit(self.X_train)
            self.X_train_re = self.re_model.transform(self.X_train)
            self.X_test_re = self.re_model.transform(self.X_test)
        elif self.choice == "mds+svm":
            self.re_model = MDS(n_components=n_components, max_iter=100)
            self.X_train_re = self.re_model.fit_transform(self.X_train)
            self.X_test_re = self.re_model.fit_transform(self.X_test)
        else:
            self.X_train_re = self.X_train
            self.X_test_re = self.X_test
        self.re_time = time() - t0
        print(f"{choice} 降维时间: {self.re_time:.3f}s")
    """功能是返回降维后的训练集数据 (self.X_train_re)"""
    def get_X_train_re(self):
        return self.X_train_re
    """返回降维后的测试集数据：
self.X_test_re 存储的是通过指定降维方法（如 PCA、KPCA 等）处理后的测试集特征。"""
    def get_X_test_re(self):
        return self.X_test_re

# SVM 类
"""定义了一个名为 svm 的类，主要用于支持向量机 (SVM) 模型的训练、预测和评估"""
"""输入参数：
dataset：my_dataset 的实例，提供训练和测试数据。
decomposition：my_decomposition 的实例，用于获取降维后的训练集和测试集。
kernel：指定 SVM 的核函数类型，默认使用径向基核 (rbf)。
class_weight：控制类别权重，'balanced' 会根据类别样本的数量自动调整权重。
功能：
初始化 SVM 类，加载降维后的训练集 (self.X_train_re) 和测试集 (self.X_test_re)。
加载训练集标签 (self.y_train) 和测试集标签 (self.y_test)。"""
class svm:
    def __init__(self, dataset, decomposition, kernel='rbf', class_weight='balanced'):
        self.dataset = dataset
        self.decomposition = decomposition
        self.X_train_re = self.decomposition.get_X_train_re()
        self.X_test_re = self.decomposition.get_X_test_re()
        self.y_train, self.y_test = self.dataset.get_data()[2:]
    """功能：
使用支持向量机 (SVM) 模型进行训练。
SVC 是 scikit-learn 中的支持向量机分类器。
fit() 方法将降维后的训练数据和对应标签传入，完成模型的拟合。
存储结果：
训练好的 SVM 模型存储在 self.clf 属性中。"""
    def my_fit(self):
        self.clf = SVC(kernel='rbf', class_weight='balanced').fit(self.X_train_re, self.y_train)
    """使用训练好的 SVM 模型对降维后的测试集进行预测。
predict() 方法返回测试集中每个样本的预测类别标签。
"""
    def my_predict(self):
        return self.clf.predict(self.X_test_re)
    """使用 classification_report 函数生成详细的分类评估报告。
包括每个类别的精度 (precision)、召回率 (recall)、F1 分数 (f1-score) 和支持样本数 (support)。
使用 self.dataset.target_names 映射标签到人名或类别名"""
    def model_report(self, y_pred):
        print(classification_report(self.y_test, y_pred, target_names=self.dataset.target_names))

# 绘制宏平均 ROC 曲线
"""功能是绘制宏平均 ROC 曲线，用于评估分类模型在多类别分类任务中的整体性能
ROC 曲线 (Receiver Operating Characteristic Curve) 是一种评估分类模型性能的工具，显示不同阈值下模型的真正例率 (TPR) 和假正例率 (FPR)。
宏平均 (Macro-Averaged) 是一种多类别评价方法，将每个类别的 ROC 曲线计算后取平均，以衡量模型的整体性能。
AUC (Area Under Curve) 是 ROC 曲线下的面积，表示模型区分能力的大小。AUC 值越接近 1，模型性能越好。"""

# def plot_Macro_roc(dataset, choices, components=150):
#     n_classes = dataset.n_classes
#     plt.figure(figsize=(10, 10))
#     for choice in choices:
#         decomposition = my_decomposition(dataset, choice, components)
#         model = svm(dataset, decomposition)
#         model.my_fit()
#         y_test = label_binarize(model.y_test, classes=np.arange(n_classes))
#         X_test = decomposition.get_X_test_re()
#         y_scores = model.clf.decision_function(X_test)
#         fpr, tpr, _ = roc_curve(y_test.ravel(), y_scores.ravel())
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f"{choice} (AUC={roc_auc:.3f})")
#     plt.plot([0, 1], [0, 1], '--', color='gray', lw=2)
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Macro-Averaged ROC Curve')
#     plt.legend(loc='lower right')
#     plt.show()
def plot_Macro_roc(dataset, choices, components=150):
    n_classes = dataset.n_classes
    plt.figure(figsize=(10, 10))
    for choice in choices:
        decomposition = my_decomposition(dataset, choice, components)
        model = svm(dataset, decomposition)
        model.my_fit()
        y_test = label_binarize(model.y_test, classes=np.arange(n_classes))
        X_test = decomposition.get_X_test_re()
        y_scores = model.clf.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_scores.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{choice} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], '--', color='gray', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-Averaged ROC Curve')
    plt.legend(loc='lower right')

    # 新增：保存图像
    save_path = "./roc_macro.png"
    plt.savefig(save_path, dpi=300)
    print(f"ROC 曲线图已保存到：{save_path}")

    # 不再显示图像，避免卡顿
    plt.close()

# 主函数 已经可以正常处理两个数据集了
if __name__ == "__main__":
  # 用户选择数据集类型
    data_type = "lfw"  # "lfw" 或 "orl"
    data_path = "./lfw_dataset/" if data_type == "lfw" else "./orl_dataset/"

    # 数据集加载
    dataset = my_dataset(dataset_type=data_type, data_path=data_path, min_faces_per_person=10, resize=0.4)

    #降维方法与参数
    choices = ["pca+svm", "kpca+svm", "lle+svm", "isomap+svm", "mds+svm"]
    #choices = ["pca+svm", "kpca+svm"]
    components = 150
    # 绘制宏平均 ROC 曲线
    plot_Macro_roc(dataset, choices, components)
    print(">>>>>>>>>>>>>>开始使用 PCA 进行训练与测试啦>>>>>>>>>>>>")
    # 使用 PCA 进行训练与测试
    decomposition = my_decomposition(dataset, "pca+svm", components)
    model = svm(dataset, decomposition)
    model.my_fit()
    y_pred = model.my_predict()
    model.model_report(y_pred)
