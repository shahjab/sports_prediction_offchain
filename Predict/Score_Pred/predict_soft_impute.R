library('softImpute')

df.pace = read.csv('paco.csv')
df.OR = read.csv('OR.csv')

df.pace[df.pace == 0] <- NA
df.OR[df.OR==0] = NA

df.pace = as.matrix(df.pace)
df.OR = as.matrix(df.OR)

df.pace = df.pace[,seq(2, 31)]
df.OR = df.OR[,seq(2, 31)]

class(df.pace) = 'numeric'
class(df.OR) = 'numeric'

fits.pace = softImpute(x=df.pace,trace=TRUE, type="svd", lambda=25, rank.max = 10)
pace = fits.pace$u %*% diag(fits.pace$d) %*% t(fits.pace$v)

fits.OR=softImpute(x=df.OR,trace=TRUE, type="svd", lambda=50, rank.max = 10)
OR = fits.OR$u %*% diag(fits.OR$d) %*% t(fits.OR$v)

predictions = OR * pace / 100

colnames(predictions) = colnames(df)
rownames(predictions) = colnames(df)

write.csv(predictions, 'predictions.csv')
