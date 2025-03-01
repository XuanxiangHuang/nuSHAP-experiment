# nushap -- rigorous SHAP scores
**nushap** is a tool for computing the SHAP scores from a sample of the feature space, in a model-agnostic fashion.### Synopsis**nushap** implements the computation Shapley values in XAI, using a novel characteristic function. Whereas the scores computed by the original SHAP tool can mislead human decision-makers, this is not the case with **nushap**. The characteristic function used in **nushap** is studied in the following paper:

Olivier Letoffe, Xuanxiang Huang, Joao Marques-Silva:
On Correcting SHAP Scores. CoRR abs/2405.00076 (2024)

The paper above proposes using a test for a weak AXp to decide the value of the characteristic function. In contrast with the definition of weak AXp in formal explanations, **nushap** uses sample-based explanations, which in the case of complete truth-tables represent a true weak AXp.

The case of sample-based explanations where the sample does not match feature space is studied in the following paper:

Martin C. Cooper, Leila Amgoud: Abductive Explanations of Classifiers Under Constraints: Complexity and Properties. ECAI 2023: 469-476

The case when the sample corresponds to feature space is studied in the following paper:

Xuanxiang Huang, Joao Marques-Silva: The Inadequacy of Shapley Values for Explainability. CoRR abs/2302.08160 (2023)
### Running the toolTo run the **nushap** tool, and obtain the help:% ./tools/nushap/nushap.py --helpThe tool accepts datasets in CSV format. Besides a dataset, the tool expects an instance, that must also be represented in CSV format.Consider the example from Fig 1a of the following paper:Joao Marques-Silva, Xuanxiang Huang: Explainability Is Not a Game. Commun. ACM 67(7): 66-75 (2024)The CSV file for dataset representing feature space is:<br>x1,x2,x3,target<br>0,0,0,0<br>0,0,1,4<br>0,0,2,0<br>0,1,0,0<br>0,1,1,7<br>0,1,2,0<br>1,0,0,1<br>1,0,1,1<br>1,0,2,1<br>1,1,0,1<br>1,1,1,1<br>1,1,2,1<br>and the instance file is:<br>x1,x2,x3,target<br>1,1,2,1<br>To run the **nushap** tool, a possible command is:% /tools/nushap/nushap.py --dataset DATAFILE.csv --instfile INSTFILE.csv --error 0.0025 --alpha 0.025The output produced by the **nushap** tool is:Copyright (C) 2024 Joao Marques-Silva

This program is private and unlicensed. This program may not
be modified. No parts of this program may be redistributed.
For additional restrictions, see the file NOTICES.md.

\#\#\# Svs: 1.0000 0.0000 0.0000Observe that these SHAP scores correspond exactly to what one should expect, as discussed in the following paper:

Joao Marques-Silva, Xuanxiang Huang: Explainability Is Not a Game. Commun. ACM 67(7): 66-75 (2024)More details can be obtained by increasing the verbosity of the tools, or by activating debug mode.### Notice of useThe tool is to be used as is. No changes to the code will be considered. Suggestions and bug reports should be communicated to the author by email (see below).##### Author: Joao Marques-Silva (jpms@icrea.cat)