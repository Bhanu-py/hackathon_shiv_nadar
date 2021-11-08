library(dplyr)
library(glmnet)
library(ggplot2)
library(plotROC)
library(ggcorrplot)


df <- read.csv("df2.csv", header = TRUE)
head(df)
dim(df)
summary(df)
table(df$Item_Type)

str(df)
# df$Sales <- scale(df$Sales, center=TRUE, scale = TRUE)



boxplot(df$Item_MRP)

q <- quantile(df$Item_MRP, probs = c(0.25, 0.75))
q[2]-q[1]

lr <- q[1] - 1.5 * IQR(df$Item_MRP)
ur <- q[1] + 1.5 * IQR(df$Item_MRP)


df <- subset(df, subset=df$Item_MRP > lr & df$Item_MRP< ur)
dim(df)
write.csv(df, file = 'df.csv')
# Converting chr variables to factors

df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], as.factor)
str(df)
dim(df)
# df$age <- (2021 - df$Outlet_Year)

# head(select(df, -Sales))
# svdX <- svd(select(df, -Sales))
# 
# k <- 30
# Uk <- svdX$u
# Dk <- diag(svdX$d)
# Zk <- Uk%*%Dk
# Y <- df %>%
#   pull(Sales)
# 
# m4 <- lm(Y~Zk[,1:26])
# summary(m4)


head(df)
set.seed(2021)
X <- select(df, -c(Sales))
Y <- df$Sales
head(cbind(X, Y))
# cor(cbind(X, Y))

n <- nrow(df)
n
ntrain <- round(0.7 * n)
ntrain
indtrain <- sample(1:n, ntrain)

length(indtrain)

XTrain <- X[indtrain,]
YTrain <- Y[indtrain]
XTest <- X[-indtrain,]
YTest <- Y[-indtrain]
length(YTest)
head(XTrain)
length(Y)

table(XTrain$Item_Type)



############################## The Ridge #################################################


ridge <- glmnet(
  x = XTrain,
  y = YTrain,
  alpha = 0,         # ridge: alpha = 0
  family = 'gaussian')

plot(ridge, xvar = "lambda")

cv_ridge <- cv.glmnet(
  x = as.matrix(XTrain),
  y = YTrain,
  alpha = 0,               # ridge: alpha = 0
  family= "gaussian",
  )
# 
plot(cv_ridge)
# 
sort(colnames(XTest)) == sort(colnames(testdf))
dfRidgeOpt <- data.frame(
  pred = predict(cv_ridge,
               newx = as.matrix(XTest),
               s = cv_ridge$lambda.min,
               type = "response") %>% c(.),
  actual = YTest)
head(dfRidgeOpt)
sqrt(mean((dfRidgeOpt$actual - dfRidgeOpt$pred)^2))


testdf <- read.csv('testdf2.csv')

head(testdf)
head(XTest)
dim(testdf)
# testdf$age <- (2021 - testdf$Outlet_Year)
# testdf <- select(testdf, -Outlet_Year)



predicted <- data.frame(
  Sales = predict(cv_ridge,
                 newx = as.matrix(testdf),
                 s = cv_ridge$lambda.min,
                 type = "response"))

colnames(predicted) <- "Sales"

head(predicted)

write.csv(predicted, 'data/submission.csv')


############################## The Lasso #################################################



lasso <- glmnet(
  x = XTrain,
  y = YTrain,
  alpha = 1,         # lasso: alpha = 1
  family="gaussian")  

plot(lasso, xvar = "lambda")


cv_lasso <- cv.glmnet(
  x = as.matrix(XTrain),
  y = YTrain,
  alpha = 1,               # lasso: alpha = 1
  type.measure = "mse",
  family = "gaussian")  

plot(cv_lasso)

cv_lasso$lambda.min
dfLassoOpt <- data.frame(
  pred = predict(cv_lasso,
               newx = as.matrix(XTest),
               s = cv_lasso$lambda.min,
               type = "response") %>% c(.),
  actual = YTest)


head(dfLassoOpt)
sqrt(mean((dfLassoOpt$actual - dfLassoOpt$pred)^2))


predicted <- data.frame(
  Sales = predict(cv_lasso,
                  newx = as.matrix(testdf),
                  s = cv_lasso$lambda.min,
                  type = "response"))

colnames(predicted) <- "Sales"

head(predicted)

write.csv(predicted, 'submission.csv')

############################## The Elastic Net #################################################

mElastic  <- glmnet(
  x = XTrain,
  y = YTrain,
  alpha = 0.5,         # lasso: alpha = 1
  family="gaussian")  

plot(mElastic, xvar = "lambda")


mCvElastic  <- cv.glmnet(
  x = as.matrix(XTrain),
  y = YTrain,
  alpha = 0.5,               # lasso: alpha = 1
  type.measure = "mse",
  family = "gaussian")  

plot(mCvElastic)

mCvElastic$lambda.min
dfElast  <- data.frame(
  pred = predict(mElastic,
                 newx = as.matrix(XTest),
                 s = mCvElastic$lambda.min,
                 type = "response") %>% c(.),
  actual = YTest)


head(dfElast)
sqrt(mean((dfElast$actual - dfElast$pred)^2))


predicted <- data.frame(
  Sales = predict(mElastic,
                  newx = as.matrix(testdf),
                  s = mCvElastic$lambda.min,
                  type = "response"))

colnames(predicted) <- "Sales"

head(predicted)

write.csv(predicted, 'submission.csv')


##################### Linear Model #########################################################


# opt_data1 <- data.frame(YTrain, XTrain[, 1:ncol(XTrain)])
# sd<- opt_data1
# sd[sapply(opt_data1, is.numeric)] <- scale(opt_data1[sapply(opt_data1, is.numeric)])
# 
# head(opt_data1)
# dim(opt_data1)
# 
# 
# model1 <- lm(YTrain ~ ., data=opt_data1)
# mean(model1$residuals^2)
# 
# predict(model1)
# 
# data <- data.frame(pred = predict(model1), actual = YTrain)
# sqrt(mean((data$actual - data$pred)^2))
# summary(model1)
# 
# #################### Model 2 (without Item_Type) #######################
# opt_data2 <- select(data.frame(YTrain, XTrain[, 1:ncol(XTrain)]), -Item_Type)
# head(opt_data2)
# dim(opt_data2)
# 
# model2 <- lm(YTrain ~ ., data=opt_data2)
# summary(model2)
# 
# 
# model.matrix(~0+., data=opt_data2) %>% 
#   cor(use="pairwise.complete.obs") %>% 
#   ggcorrplot(show.diag = F, type="lower", lab=TRUE, lab_size=2)
# 
# 
# mean(model1$residuals ** 2)







# 
# eval_results <- function(true, predicted, df) {
#   SSE <- sum((predicted - true)^2)
#   SST <- sum((true - mean(true))^2)
#   R_square <- 1 - SSE / SST
#   RMSE = sqrt(SSE/nrow(df))
#   
#   
#   # Model performance metrics
#   data.frame(
#     RMSE = RMSE,
#     Rsquare = R_square
#   )

# sqrt(sum((dfRidgeOpt$pred - dfRidgeOpt$actual) ** 2 / nrow(dfRidgeOpt)))


# eval_metrics = function(model, df, predictions, target){
#   resids = df[,target] - predictions
#   resids2 = resids**2
#   N = length(predictions)
#   r2 = as.character(round(summary(model)$r.squared, 2))
#   adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
#   print(adj_r2) #Adjusted R-squared
#   print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
# }
# 
# eval_metrics(cv_ridge, df[-indtrain,], dfRidgeOpt$pred, 'Sales')

