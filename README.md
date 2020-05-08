# Image-Retrieval-Using-Locality-Sensitive-Hashing-Algorithm

Locality Sensitive Hashing(LSH) is Nearest Neighbor Search algorithm that primarily targeting large dataset with high dimensions. The problem in existing algorithm is to find the nearest neighbor in high dimensional datasets with less processing time. This issue often occurs in recommender system applications, data mining applications, etc. Linear search can be useful when dealing with low dimensional dataset. However, it is time consuming when dealing with high dimensional dataset. In order to solve this problem, a particular type of hash functions was designed to project two similar datasets to one hash value. This method is called LSH. 

There are some challenges existing in the current LSH algorithm. For example, when we apply the LSH algorithm for applications such as finding similarities between different documents, we have to rebuild the hash table every time we add a document. This process is very costly and inefficient. Our objective is to improve the efficiency of the algorithm by applying bitmap indexing. 

In this project, we propose a new LSH-based algorithm to improve hashing techniques and compression performance. Our approach uses bitmap indexing to speed up and simplify the application of LSH algorithm. We will compare the results from our approach with existing LSH algorithm. The final outcome will be evaluated based on processing time with different types of datasets.

# What is LSH?
LSH is a hashing based algorithm to identify approximate nearest neighbors. In the normal nearest neighbor problem, there are a bunch of points (let's refer to these as training set) in space and given a new point, objective is to identify the point in training set closest to the given point. Complexity of such process is linear [for those familiar with Big-O notation, O(N), where N is the size of training set]. An approximate nearest neighboring algorithm tries to reduce this complexity to sub-linear (less than linear but can be anything). Sub-linear complexity is achieved by reducing the number of comparisons needed to find similar items.

LSH works on the principle that if there are two points in feature space closer to each other, they are very likely to have same hash (reduced representation of data). LSH primarily differs from conventional hashing (aka cryptographic) in the sense that cryptographic hashing tries to avoid collisions but LSH aims to maximize collisions for similar points. In cryptographic hashing a minor perturbation to the input can alter the hash significantly but in LSH, slight distortions would be ignored so that the main content can be identified easily. The hash collisions make it possible for similar items to have a high probability of having the same hash value.

> Locality Sensitive Hashing (LSH) is a generic hashing technique that aims, as the name suggests, to preserve the local relations of the data while significantly reducing the dimensionality of the dataset.

Now that we have established LSH is a hashing function that aims to maximize collisions for similar items, let's formalize the definition:

A hash function h is Locality Sensitive if for given two points a, b in a high dimensional feature space,

> * Pr(h(a) == h(b)) is high if a and b are near
> * Pr(h(a) == h(b)) is low if a and b are far
> * Time complexity to identify close objects is sub-linear

## Bucketed Random Projection for Euclidean Distance
Bucketed Random Projection is an LSH family for Euclidean distance. The Euclidean distance is defined as follows:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>d</mi>
  <mo stretchy="false">(</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo>,</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi mathvariant="bold">y</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msqrt>
    <munder>
      <mo>&#x2211;<!-- ∑ --></mo>
      <mi>i</mi>
    </munder>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>x</mi>
      <mi>i</mi>
    </msub>
    <mo>&#x2212;<!-- − --></mo>
    <msub>
      <mi>y</mi>
      <mi>i</mi>
    </msub>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
  </msqrt>
</math>

Its LSH family projects feature vectors x onto a random unit vector v and portions the projected results into hash buckets:


<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>h</mi>
  <mo stretchy="false">(</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mrow class="MJX-TeXAtom-ORD">
    <mo maxsize="1.623em" minsize="1.623em">&#x230A;</mo>
  </mrow>
  <mfrac>
    <mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mi mathvariant="bold">x</mi>
      </mrow>
      <mo>&#x22C5;<!-- ⋅ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi mathvariant="bold">v</mi>
      </mrow>
    </mrow>
    <mi>r</mi>
  </mfrac>
  <mrow class="MJX-TeXAtom-ORD">
    <mo maxsize="1.623em" minsize="1.623em">&#x230B;</mo>
  </mrow>
</math>
where r is a user-defined bucket length. The bucket length can be used to control the average size of hash buckets (and thus the number of buckets). A larger bucket length (i.e., fewer buckets) increases the probability of features being hashed to the same bucket (increasing the numbers of true and false positives).


# Prerequisites (Local)
* Jupiter Notebook
* Pyhton3
* pySpark

# How to run it?
## Step 1. Clone the Repo
 ```
 git clone git@github.com:songruizhe/LSH-Algorithm-Improvement-By-Applying-Bitmap-Indexin.git
 cd LSH-Algorithm-Improvement-By-Applying-Bitmap-Indexing
 ```
## Step 2. Open Jupyter Notebook in CLI
 ```
 jupyter Notebook
 ```

## Step 3. Open MasterProject.ipynb and run line by line
if there is some libarary/module missing, pip install the corresponding library/module.
