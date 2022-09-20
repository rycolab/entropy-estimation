from setuptools import setup


install_requires = [
	"pandas",
    "ndd",
    "plotnine",
    "scikit-misc",
    "scipy",
    "matplotlib",
	"conllu"
]


setup(
	name="entropy",
	install_requires=install_requires,
	version="0.1",
	scripts=[],
	packages=['entropy']
)
