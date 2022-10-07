################################################################################
#                             Mini Projeto
#
#   Prevendo a Inadimplência de Clientes com Machine Learning e Power BI
#
################################################################################


setwd("B:/R_DOCUMENTS/RFundamentos/FirstProject")
getwd()

install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

dados_clientes <- read.csv("dataset.csv")

View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)


#################  Análise Exploratória, Limpeza e Transformação ###############

# removendo a coluna ID

dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

# renomeando a coluna de classe

colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# verificando valores ausentes e removendo do dataset

sapply(dados_clientes, function(x) sum(is.na(x)))
?missmap
missmap(dados_clientes, main = "valores ausentes observados")
dados_clientes <- na.omit(dados_clientes)

# convertendo os atributos genero, escolaridade, estado civil e idade
# para fatores(categorias)

# renomeando colunas categóricas

colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)

# genero

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero,
                      c(0,1,2),
                      labels = c("Masculino",
                                 "Feminino"))

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
View(dados_clientes)

# escolaridade

str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c("Pos Graduado",
                                              "Graduado",
                                              "Ensino Medio",
                                              "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

# estado civil

str(dados_clientes$Estado_Civil) 
summary(dados_clientes$Estado_Civil) 
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil, 
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))
View(dados_clientes$Estado_Civil) 
str(dados_clientes$Estado_Civil) 
summary(dados_clientes$Estado_Civil) 

# convertendo a variável para o tipo fator com faixa etária

str(dados_clientes$Idade) 
summary(dados_clientes$Idade) 
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade, 
                            c(0,30,50,100), 
                            labels = c("Jovem", 
                                       "Adulto", 
                                       "Idoso"))
View(dados_clientes$Idade) 
str(dados_clientes$Idade) 
summary(dados_clientes$Idade)
View(dados_clientes)

# convertendo a variavel que indica pagamentos para o tipo fator

str(dados_clientes)
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# lidando com valores ausentes no dataset após conversões

sapply(dados_clientes, function(x) sum(is.na(x)))
dados_clientes <- na.omit(dados_clientes)

sapply(dados_clientes, function(x) sum(is.na(x)))


# alterando a variável dependente para o tipo fator

str(dados_clientes$Inadimplente)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)

str(dados_clientes$Inadimplente)

# total de inadimplentes versus não-inadimplentes

table(dados_clientes$Inadimplente)

# porcentagens entre as classes

prop.table(table(dados_clientes$Inadimplente))

# plot da distribuição usando ggplot2

qplot(Inadimplente, data = dados_clientes, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# set seed

set.seed(12345)

# amostragem estratificada
# seleciona as linhas de acordo com a variável inadimplente como strata

indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

# definimos os dados de treinamento como subconjunto do conjunto de dados original
# com números de indice de linha (conforme identificado acima) e todas as colunas

dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)
dim(dados_treino)


# comparamoos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)), 
                       prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# tudo o que não está no dataset de treino, estará no dataset de teste
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)


######################### Modelo Machine Learning ##############################

# construindo a primeira versão do modelo

modelo_v1 <- randomForest::randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

# avaliando o modelo

plot(modelo_v1)

# previsões com dados de teste

previsoes_v1 <- predict(modelo_v1, dados_teste)

# confusion matrix

?caret::confusionMatrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive = "1")
cm_v1

# balanceamento da classe

install.packages("ROSE")
library(ROSE)

barplot(prop.table(table(dados_treino$Inadimplente)),
        col = rainbow(2),
        ylim = c(0, 0.7),
        main = "Class Distribution")

prop.table(table(dados_treino$Inadimplente))

dados_treino_bal <- ovun.sample(Inadimplente~., data = dados_treino, method = "over")$data
table(dados_treino_bal$Inadimplente)


barplot(prop.table(table(dados_treino_bal$Inadimplente)),
        col = rainbow(2),
        ylim = c(0, 0.7),
        main = "Class Distribution")

prop.table(table(dados_treino_bal$Inadimplente))

# construindo a segunda versão do modelo

modelo_v2 <- randomForest::randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

plot(modelo_v2)

previsoes_v2 <- predict(modelo_v2, dados_teste)

cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive = "1")
cm_v2


# verificando importância das variáveis preditoras para as previsões

View(dados_treino_bal)
randomForest::varImpPlot(modelo_v2)

# construindo a terceira versão do modelo apenas com as variáveis mais importantes

modelo_v3 <- randomForest::randomForest(Inadimplente ~ PAY_0 + BILL_AMT1 + LIMIT_BAL + PAY_AMT2 + 
                                          PAY_AMT1 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + 
                                          BILL_AMT6 + PAY_AMT3 + PAY_AMT6 + PAY_AMT4 + 
                                          PAY_AMT5, data = dados_treino_bal)

modelo_v3

plot(modelo_v3)

previsoes_v3 <- predict(modelo_v3, dados_teste)

cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive = "1")
cm_v3

# salvando os objetos R

saveRDS(dados_treino_bal, file = "dados_treino_bal.rds")
saveRDS(modelo_v3, file = "modelo_v3.rds")


# carregando os objetos R

modelo_final <- readRDS("modelo_v3.rds")
treino_bal <- readRDS("dados_treino_bal.rds")


# previsões com dados de 3 novos clientes

# dados dos clientes
PAY_0 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280) 
LIMIT_BAL <- c(5000, 10000, 15000)
PAY_AMT2 <- c(1500, 1300, 1150) 
PAY_AMT1 <- c(1100, 1000, 1200) 
BILL_AMT3 <- c(400, 300, 350)
BILL_AMT4 <- c(500, 400, 320)
BILL_AMT5 <- c(450, 600, 500)
BILL_AMT6 <- c(200, 500, 440)
PAY_AMT3 <- c(800, 900, 1500)
PAY_AMT6 <- c(1000, 2000, 900)
PAY_AMT4 <- c(1300, 1000, 1100)
PAY_AMT5 <- c(1200, 1330, 950)

# concatena em um dataframe

novos_clientes <- data.frame(PAY_0, BILL_AMT1, LIMIT_BAL, PAY_AMT2, 
                               PAY_AMT1, BILL_AMT3, BILL_AMT4, BILL_AMT5, 
                               BILL_AMT6, PAY_AMT3, PAY_AMT6, PAY_AMT4, 
                               PAY_AMT5)
View(novos_clientes)

str(treino_bal)
str(novos_clientes)

# convertendo os tipos de dados com o mesmo tipo dos dados de treino

novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(treino_bal$PAY_0))
str(novos_clientes)


# primeira previsão

previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
previsoes_novos_clientes



# outro exemplo de previsão

# dados dos clientes
PAY_0 <- c(2, 0, 0)
BILL_AMT1 <- c(14000, 720, 880) 
LIMIT_BAL <- c(2000, 10000, 30000)
PAY_AMT2 <- c(15000, 1300, 1150) 
PAY_AMT1 <- c(1200, 1000, 1200) 
BILL_AMT3 <- c(16000, 300, 350)
BILL_AMT4 <- c(16893, 600, 320)
BILL_AMT5 <- c(18200, 100, 500)
BILL_AMT6 <- c(17769, 500, 440)
PAY_AMT3 <- c(1284, 300, 1500)
PAY_AMT6 <- c(644, 2000, 900)
PAY_AMT4 <- c(1591, 1000, 1600)
PAY_AMT5 <- c(0, 1330, 950)



novos_clientes_2 <- data.frame(PAY_0, BILL_AMT1, LIMIT_BAL, PAY_AMT2, 
                             PAY_AMT1, BILL_AMT3, BILL_AMT4, BILL_AMT5, 
                             BILL_AMT6, PAY_AMT3, PAY_AMT6, PAY_AMT4, 
                             PAY_AMT5)

View(novos_clientes_2)

# convertendo os tipos de dados com o mesmo tipo dos dados de treino

novos_clientes_2$PAY_0 <- factor(novos_clientes_2$PAY_0, levels = levels(treino_bal$PAY_0))
str(novos_clientes)

# segunda previsão

previsoes_novos_clientes_2 <- predict(modelo_final, novos_clientes_2)
previsoes_novos_clientes_2
















































