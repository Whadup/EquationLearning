##  Searching for Scientific Publications

Every day, huge amounts of new scientific manuscripts are published. On the pre-print service arxiv.org alone, more than x papers where uploaded last year. This flood of scientific manuscripts is impossible to filter, index and organize manually. Hence scientist rely heavily on existing search engines like Google Scholar or Mendeley to find relevant content, mainly by using keyword search.

Keyword search is tricky, because similar concepts are referred to using different terminology between disciplines or even between subfields. For instance in the machine learning community we talk about features and labels, wheres statisticians usually refer to the same concepts as independent and dependent variables.

Instead, we propose to organize publications not by their plain-text content, but by the equations used to describe their respective ideas. This way we hope to find connections between fields that do not share a common vocabulary. Examples may include connections between probabilistic graphical models used in artificial intelligence and models of non-standard physics.



## Problem

We want to provide a search-engine that allows users to find equations related to their query equations. These relations shall be judged by a machine learning model that assesses the similarity of a given pair of equations. Then the search engine can lookup responses to a given query using an efficient index structure.

Training such a machine learning system usually requires the use of logged user interactions with the search engine: When a user had the following query, the system offered a set of possible results and the user interacted with a number of them. However, we do not have a running system; we suffer from the well-known cold-start problem. All we have is a collection of documents, each document containing a number of equations. We propose a number of different training tasks for training a model without logged user interactions.

We assemble an evaluation data set, where we hand-label a small number of equations and assign them to groups based on domain knowledge. A successful system should judge the similarities within those groups as high and between groups as low. Note that these gold-labels are not used during training.

## The Dataset

We have downloaded ~25,000 publications that provide not only a pdf file, but also the LaTex sources used to generate the pdf. From these sources we extracted maths environments. Of these snippets, were able to compile more than 660,000 equations; the major cause for failed compilation being more compicated user-defined LaTex macros.



​    arxiv/

​       1711.11486v1/

​            abstract.tex

​            keywords.tex

​            1.png

​            1.tex



