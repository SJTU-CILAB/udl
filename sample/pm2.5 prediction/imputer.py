import pickle
from datalayer import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *
for name in ["NewYorkState_pm2.5","NewYorkState_pop","NewYorkState_light","Shanghai_pm2.5","Shanghai_pop","Shanghai_light"]:
    data = pickle.load(
            open("data/"+name+".pickle", "rb")
        )
#    data.data = IterativeImputer().fit_transform(data.data)
    data.data = SimpleImputer().fit_transform(data.data)
    pickle.dump(data,open("data/"+name+"_imputedS.pickle", "wb"))
