# LASSO - Least Absolute Shrinkage and Selection Operator
# Supervised Statistical/Machine Learning (field of study)

# Training LASSO regression in k-fold cross-validation framework (k = 10)
# Goal: improve model prediction (accuracy). Explore explanatory variables that optimize the model.

# Shrinkage
  #**# shrinks the least important coefficients to zero
  # (y - f(x))^2 + lambda * ( abs(b[1]) + abs(b[2]) + .... + abs(b[n]) )

# Hyperparameters
  # alpha - mixing percentage (constant)
  # lambda - regularization tuning parameter

# Optimize model accuracy with model parsimony

# Indentifying optimal lambda (best lambda value)
  # Root mean-squared error (RMSE)
  # R-squared, variance explained in the outcome of the model
  # Mean-squared error

# Model Type Selection
  # OLS ~ Linear LASSO Regression

#**# Cross-Validation (k-fold cross-validation)

# Predictive Modeling Framework
  #**# 80/20 random split (80% training data, 20% test data)


# --- Initial Steps

# install.packages("readr")
library(readr) 
df <- read_csv("lasso.csv")
dim(df)

# --- Statistical Assumptions

##### Splitting the Data

# install.packages("caret", dependencies = c("Depends", "Suggests"))
library(caret)
set.seed(123)

# split and create index matrix of selected values
in_train <- createDataPartition(df$y, p = 0.8, list = FALSE, times = 1)
df <- as.data.frame(df)
train_df <- df[in_train, ] # 80% obs
test_df <- df[-in_train, ] # 20% obs

# specify k-fold cross-validation (k = 10)
ctrlspecs <- trainControl(method = "cv", number = 10, 
                          savePredictions = "all")

##### Specify & Train LASSO Regression Model

lambda_vector <- 10^seq(5, -5, length = 100) # length = 500 overkill?

set.seed(123)

# Specify Lasso regression model to be estimates using training data
# and k-fold cross-validation framework

model1 <- train(y ~ .,
                data = train_df,
                preProcess = c("center","scale"),
                method = "glmnet",
                tuneGrid = expand.grid(alpha = 1, lambda = lambda_vector),
                trControl = ctrlspecs,
                na.action = na.omit
                )
plot(model1)

# Best (optimal) tuning parameter lambda in terms of root mean squared error of the model. 
# Average best lambda value across folds of cross-validation
model1$bestTune
model1$bestTune$lambda

# LASSO regression model coefficients (parameter estimates)
coef(model1$finalModel, model1$bestTune$lambda)


plot(log(model1$results$lambda),
     model1$results$RMSE,
     xlab = "log(lambda)",
     ylab = "RMSE",
     xlim = c(-5, 0)
     )

plot(log(model1$results$lambda),
     model1$results$Rsquared,
     xlab = "log(lambda)",
     ylab = "R-squared",
     xlim = c(-5, 0)
)

# Variable importance 
# getModelInfo("glmnet")$glmnet$varImp
# varImp looks at the final coefficients of the fit and then takes the absolute value to rank the coefficients. 
varImp(model1)

# Data visualization of variable importance
# install.packages("ggplot2")
library(ggplot2)
ggplot(varImp(model1))

##### Model Prediction

predictions1 <- predict(model1, newdata = test_df)

# Model performance/accuracy (in-built RMSE & R-Squared formulas: RMSE = sqrt(mean((test_df$y - predictions1)^2)) & RSQ = 1 - SSE/SST where SSE = sum((predictions1 - test_df$y)^2) and SST = sum((test_df$y - mean(test_df$y))^2)
model1_performance <- postResample(pred = predictions1, obs = test_df$y)   

##### Compare OLS Multiple Linear Regression to Lasso Regression Model

set.seed(123)

# Specify OLS MLR model to be estimated using train_df k-fold cross-validation
model2 <- train(y ~ .,
                   data = train_df,
                   preProcess = c("center", "scale"),
                   method = "lm",
                   trControl = ctrlspecs,
                   na.action = na.omit)

# summary(lm(y ~ ., data = train_df))

model_list <- list(model1, model2)
resamples <- resamples(model_list)
summary(resamples)

# Compare models using paired-samples (one-sample) t-test
compare_models(model1, model2, metric = "RMSE")
compare_models(model1, model2, metric = "Rsquared")

# Predict outcome, using model from training data, based on test data
predictions2 <- predict(model2, newdata = test_df)

# Model performance
model2_performance <- postResample(pred = predictions2, obs = test_df$y)

# Compare model1 (= LASSO) and model2 (= OLS MLR) predictive performance based on test_df
round(model1_performance, 3)
round(model2_performance, 3)


# --- Visualizations combining glmnet, plotmo & caret

# install.packages("glmnet")
# install.packages("plotmo")
library(glmnet)
library(plotmo)
set.seed(123)

# define the model equation and extract the design matrix X
X <- model.matrix(y ~ ., data = train_df)[,-1]

# and the outcome
Y <- train_df$y

# model1 w/ glmnet function
model_to_plot <- glmnet(x = X, y = Y)
cv_model_to_plot <- cv.glmnet(x = X, y = Y, alpha = 1) 

##### Needed Glmnet Plots

# MSE and lambda
plot(cv_model_to_plot)
# Trace plot
plot(cv_model_to_plot$glmnet.fit, "lambda", label=T)

##### Some Bonus Colorful Sandbox Trials for Trace Plots

# or use glm_coef <- coef(model_to_plot, cv_model_to_plot$lambda.min)
glm_coefs <- coef(model1$finalModel, model1$bestTune$lambda)

coef_increase <- dimnames(glmcoefs[glmcoefs[ ,1] > 0, 0])[[1]]
coef_decrease <- dimnames(glmcoefs[glmcoefs[ ,1] < 0, 0])[[1]]

#get ordered list of variables as they appear at smallest lambda
allnames <- names(coef(model1$finalModel)[,
                          ncol(coef(model1$finalModel))][order(coef(model1$finalModel)[,
                                                          ncol(coef(model1$finalModel))],decreasing=TRUE)])

#remove intercept
allnames <- setdiff(allnames,allnames[grep("Intercept",allnames)])

#assign colors
cols <- rep("gray",length(allnames))
cols[allnames %in% coef_increase] <- "green"     
cols[allnames %in% coef_decrease] <- "pink"   

# trace plot
plot_glmnet(model1$finalModel, label = TRUE, s = model1$bestTune$lambda, col = cols)

# create a function to transform coefficient of glmnet and cvglmnet to data.frame
coeff2dt <- function(fitobject, s) {
  coeffs <- coef(fitobject, s) 
  coeffs.dt <- data.frame(name = coeffs@Dimnames[[1]][coeffs@i + 1], coefficient = coeffs@x) 
  
  # reorder the variables in term of coefficients
  return(coeffs.dt[order(coeffs.dt$coefficient, decreasing = T),])
}

coeffs.table <- coeff2dt(fitobject = cv_lambda_lasso, s = "lambda.min")

ggplot(data = coeffs.table) +
  geom_col(aes(x = name, 
               y = coefficient, 
               fill = {coefficient > 0})) +
  xlab(label = "") +
  ggtitle(expression(paste("Lasso Coefficients with ", lambda, " = ..."))) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")


