from distutils.core import setup

NAME = "CSO-RF_FeatureSelection"

DESCRIPTION = "Feature Selection Repository in Python"

KEYWORDS = "Feature Selection Repository"

AUTHOR = "Kleyson Morais de Sousa"

AUTHOR_EMAIL = "kleysonb@uft.edu.br"

URL = "https://github.com/Kleysonb/CSO-RF_FeatureSelection"

VERSION = "1.0.0"

setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    keywords = KEYWORDS,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    url = URL,
    packages =["CSO", "datasets", "EvaluationMetric"] ,
)
