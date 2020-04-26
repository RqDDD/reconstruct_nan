import CompletionNaN as cn
import sys

##data_path = sys.argv[1]
data_path = "YOUR/DATA/PATH.csv"


##def eprint(*args, **kwargs):
##    print(*args, file=sys.stderr, **kwargs)



complet = cn.CompletData(data_path, 'ID', 'TARGET')

coef = complet.importantCoeffFinalTarget(number_to_keep = 12)
print(coef)
complet.fillColumns(coef)

complet.export("results.csv")
