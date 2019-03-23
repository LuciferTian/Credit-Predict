data = read.csv('new_data.csv')
View(data)
dim = dim(data)  # 1402*11
set.seed(1)
# 分层随机抽样100个样本作为测试集，其他作为训练集
attach(data)
data$Label = as.factor(data$Label)
data$APPLY_TERM_TIME = as.factor(data$APPLY_TERM_TIME)
data = data[order(Label), ]  # 按标签列排序
bad = sum(Label)  # 违约样本646，因此未违约样本756
ratio = bad / dim[1]  # 违约样本占比0.46，因此分层采样46:54
test_index_good = sample(1 : 756, 54)
test_index_bad = sample(757 : 1402, 46)
test_index = c(test_index_bad, test_index_good)
testing = data[test_index,]
testing_y = testing[,11]
testing_x = testing[,1:10]
training = data[-test_index,]

# 载入C50包
library(C50)

# winnow: 建立模型之前是否进行特征选择
# noGlobalPruning: 是否进行全局剪枝
# trials:boosting iterations(C5.0相比C4.5改进的就是引入了boosting)
C5.0tree = C5.0(Label~., data = training, trials = 100,
                control = C5.0Control(winnow = TRUE,
                noGlobalPruning = FALSE))

summary(C5.0tree)
plot(C5.0tree)

# type:可选参数，取prob时，给出预测点属于各个类别的概率；取class时，给出预测点的类别
pred = predict(C5.0tree, newdata = testing_x, type = "class") 
table(testing_y, pred)  # 生成混淆矩阵
precision = 45 / 48  # 0.93
recall = 45 / 46  #0.98
