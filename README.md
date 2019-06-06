# Data Integration

# Chapter 1: Data Integration Introduction
- DWs are **central repositories** of integrated data from one or more **disparate sources**. They store current and historical data in **one single place** that are used for creating analytical reports for workers throughout the enterprise.
```SQL
SELECT C.name FROM Students S, Takes T, Courses C
WHERE S.name ='Mary' and S.ssn=T.ssn and T.cid = C.cid
```
- The **wrappers** or **loaders** request and parse data from the **sources**. The **mediated schema** or central **data warehouse** abstracts all source data, and the user poses queries over this.
- Between the *sources* and the *mediated schema*, **source descriptions** and their associated **schema mappings**, or a set of transformations, are used to *convert the data* from the source schemas and values into the global representation.
- **Query processing** in a data integration system differs from traditional database query processing in two main ways. 
  - First, the query needs to be *reformulated* from the mediated schema to the schema of the sources.
  - Second, query execution may be *adaptive* in that the query execution plan may change as the query is being executed.
- **Mediated Schema**: 
  - Independence of 1. source & location; 2. Data Model, Syntax; 3. Semantic Variations. Make Virtual Schema on top of a database; Treat the different database as sources
- **Why is Data Integration Hard?**: 1. Systems-Level: different platforms, distributed query; 2. Logical: Schema heterogeneity; 3. Social: security and privacy.
- **Data warehousing**: It integrates by bringing the data into a single physical warehouse. It defines a procedural *mappings* in an *ETL tool* to import the data and clean it. It periodically copies all of the data from the data sources.
  - **Pros**: Queries over the warehouse don’t disrupt the data sources. üCan run very heavy-duty computations, including data mining and cleaning
  - **Cons**: Need to spend time to design the physical database layout, as well as logical. Data is generally not up-to-date (lazy or offline refresh).
- **Virtual data integration**: leave the data at the sources and access it at query time.
- **Mediation Languages**: Describes relationships between mediated schema and data sources.
- Query -> Query Reformulation -> Query Optimizer -> Execution Engine -> Wrapper of Sources

# Chapter 4: String Matching

- **Topic 1: Similarity Measures** (Accuracy)

  - **Sequence-Based**: Edit Distance, N-Wunch, Affine Gap, Smith-Waterman, Jaro, Jaro-Winkler
  - **Set-Based**: View strings as sets or sets of tokens, consider words delimited by space and q-grams
    - Overlap, Jaccard, TF/IDF
  - **Hybrid**: Generalized Jaccard, soft TF/IDF, Monge-Elkan
  - **Phonetic**: Soundex

- **Topic 2: Scaling Up String Matching**  (Scalability)

  - Inverted Index, Size / Prefix / Position / Bound Filtering

- Similartiy Measure - the higher the more likely

- Distance / Cost Measure - the smaller the more likely

- **Scalability Challenge**: Apply $s(x, y)$ to the most promising pairs

  ```python
  for each string x belongs to X:
      use FindCands to find a candidata set Z belongs to Y
      for each string y belongs to Z:
          if s(x, y) >= t:
              return (x, y) as a match pair
  ```

- **1.1. Edit / Levenshtein Distance**
  - $d(x, y)$ computes minimal cost of transforming $x$ into $y$, including *deleting*, *inserting*, and *substituting*. Note: smaller edit distance - > higher similarity.
  - e.g. $x = David Smiths$, $y = Davidd Simth$, then $d(x, y)=4$.
  - $s(x, y) = 1 - \frac{d(x, y)}{max(length(x), length(y))}$
  - Computing Edit Distance Using **Dynamic Programming**
    - Define $x = x_1x_2…x_n$, $y = y_1y_2…y_n$, and $d(i, j)$ is the edit distance between the $i_{th}$ and $j_{th}$ prefixes of $x$ and $y$.
    - Cost of dynamic programming $O(|x||y|)$.
- **1.2. Needleman-Wunch Measure**: Return the highest alignment (set of correspondence between characters) score between $x$ and $y$. alignment score = $\Sigma$SCORES of all correspondence - $\Sigma$PENALTIES of all gaps. Use a matrix to calculate scores. A match for 2, a gap for 1, a mismatch for 1.
  - Generalizes edit costs into a score matrix, insertion & deletion into gaps.
- **1.3. Affine Gap**: Previous measures consider *global alignments (i.e. match all)*. This is to find **local alignment** i.e.(**substrings**) of $x$ and $y$ that are the most similar. **Ideas**: 1. Match can restart at any position; 2. retracing values from the largest rather than the lower right. 
- **1.4. Smith-Waterman**: Mainly for comparing **short strings**
- **1.5. Jaro**: $jaro(x, y) = \frac{1}{3[c / |x| + c / |y| + (c - t/2) / c]}$. e.g. x = jon, y = ojhn. Then: x' = jon, y' = ojn; c = 3; t = 2; $jaro$ = 0.81.
- **1.6. Jaro-Winkler**: Captures cases where x and y have a **low jaro score**, but share a **prefix**. $Jaro-Winkler(x, y)=(1 - PL*PW)*Jaro(x, y)+PL*PW$. Note $PL$ is the length of the longest common prefix and the other is the weight.
- **1.7. Set Based: Overlap**: $O(x, y) = |B_x \bigcap B_y|$. *e.g.* x = dave, y = dav, $B_x$ = {#d, da, av, ve, e#}, O(x, y) = 3
- **1.8.  Set Based: Jaccard**: Considers overlapping tokens in x and y. $J(x, y) = \frac{|B_x \bigcap B_y|}{|B_x \bigcup B_y|}$ 
- **1.9. Set Based: TF/IDF**: Idea - Two strings are similar if they share distinguishing terms. Score is high if share many frequent terms.
  - $tf(t, d)$ = number of times term t apears in document d; $idf(t) = \frac{N}{N_d}$, N of all documents divided by N of documents containing t. *e.g.* x = aab, y = ac, z = a; $tf(a, x) = 2$, $idf(a) = 3/3 = 1$. 
  - Each document is converted to a vector $v_d$, and $v_d(t) = tf(t, d) * idf(t)$. *e.g.* $v_x(a) = 2$. $s(p, q) = [\Sigma {_t} v_p(t) * v_q(t)] / [\sqrt {(\Sigma {_t} v_p(t))^2} * \sqrt {(\Sigma {_t} v_{q}(t))^2}] $.
- **1.10. Generalized Jaccard**: Helps when tokens are misspelled. Find those with similarity score larger than a threshold. $GJ(x, y) = \frac{\Sigma _ {x_i, x_j} s(x_i, x_j)} {|B_x| + |B_y| - |M|}$.
- **1.11. Soft TF/IDF**: Similar to GJ, except that uses TF/IDF as the higher-level $s(x_i, x_j)$.
- **1.12. Monge-Elkan**: Break the strings into multiple substrings, and generalizes the secondary sim measures on substrings.
- **1.13. Phonetic Based: Soundex**: Map the word into a code.
- **2.1. Inverted Index Over Strings**: Converts each string y in Y into a document, builds an inverted index over these documents. Given term t, use the index to quickly find documents of Y that contain t. *Limitations*: 1. inverted lists can be long; 2. Set can be large.
- **2.2 Size Filtering**: Retrieves strings (using B-tree) in Y whose sizes make them match candidates. For **Jaccard**.
- **2.3 Prefix Filtering**: If two sets share many terms, then large subsets of them also share terms. Can do better by selecting a particular subset x' of x and checking its overlap with only a particular subset y' of y. This is for **Overlap**.
  - **Select the Subset Intelligently**: 1. reorder each x in X and y in Y in increasing
    order of their frequencies; 2. for y in Y, create its prefix y'; 3. build an inverted index over all y'; 3. for x in X, creates x', and use the above index to find all y such that x' overlaps y'.
- **2.5. Position Filtering**: Further limits the set of candidate matches by deriving an *upper bound* on the *size of overlap* between x and y. $O(x,y) >= [t/(1+t)]*(|x| + |y|)$.
- **2.6. Bound Filtering**: Used to optimize the computation of **G Jacard**. 1. For each (x,y)
  compute an upper bound UB(x,y) and a lower bound LB(x,y) on GJ(x,y). 2. If UB(x, y) <= t, (x, y) can be ignored; If LB(x, y) >= t, return (x, y) as a match. Otherwise, compute GJ(x, y).
- **2.7. Extensions**: Translate s(x,y) into constraints on a similarity measure.

# Chapter 7: Data Matching

- Why different: Can treat each tuple as a string by concatenating fields, then apply string matching techniques. But doing so is hard to apply sophisticated techniques and domain-specific knowledge

- **1. Rule-based Matching**: $sim_i$ is the sim score for the $i_{th}$ attribute of tuples x and y. Declare matched if sim crosses some threshold. For linear case, diminishing returns are not considered.

  - *Linear Weighted*: $z = sim(x, y) = \Sigma _{i} \alpha_i * sim_i(x, y)$; *Logistic*: $sim(x, y) = \frac{1} {1+e^{-z}}$.

  - **Pros**: Easy start, conceptually easy, implement, debug; Run fast; Can encode complex rules.
  - **Cons**: Labor extensive; Limited features. Hard to set weights; Unclear of the rules.

- **2. Learning- based matching**: Transform tuple $(x, y)$ into $v = <[f_1(x, y), …], 1/0>$, apply $M$ to predict whether $x$ matches $y$. Need labeled examples and find weights that minimizes the error function. DT can observe when to stop branching.

  - **Pros**: Can automatically examin many features; can construct complex rules.
  - **Cons**: Require large number of training examples. Need clustering to solve.

- **3. Matching by Clustering**: View matching tuples as the problem of constructing clusterings. The process is *iterative*. Consider distance between two points. Compare two clusters by averages, the best one / new representative record.

  - **Cons**: Hard to define $n$ clusters. Authoritative record. Computationally expensive. Hard heuristics.

- **4. Probabilistic Approaches to Matching**: Bayesian Networks, EM, Naive Bayes. With training data, can do Naive Bayes. Otherwise, construct training data with missing values and use EM to learn the missing values.

- **5. Collective Matching**: Previous matching approaches make independent matching decisions, but they are often correlated. Apears when you have *richer semantics*. **Iterative** merge things togerther based on neighborhood and look at sim of attributes. Match tuples collectively, at once and iteratively.

  - CM by clustering: *nodes* -> tuples to be matched; *edges*: relations among tuples. $sim(A, B) = \alpha * sim_{attributes}(A, B) + (1- \alpha) * sim_{neighbors}(A, B)$.

- **6. Scaling Up Matching**: 1. **Hashing**: hash into buckets and match within each bucket. will give disjoint sets; 2. **Sorting**: sort dataset using a feature, and match each tuple with only the previous $w-1$ tuples. Stronger heuristic and faster than hashing as fewer pairs required; 3. **Indexing**: use the index to quickly locate a small set of tuples that are likely to match; 4. **Canopies**: use cheap sim measure to fast group tuples into overlapping clusters, then use another sim to match tuples within each canopy; 5. **Representatives**: assign tuples into different groups, create representative for the group and match new tuple only with representative.; 6. **Parallel Processing**: hash tuples into buckets then match bucket in parallel. 7. **Combining**: hash houses into buckets using zip codes, then sort houses within each bucket using streets, then match using a sliding window.

# Chapter 5: Schema Matching

## 5.1. Problem Definition and Overview

- **Semantic Mapping**: a *query expression* that relates a schema S with a schema T. Can use *Global-as-View* (query over all sources), *Local-as-View* (semantic mapping for each table), *GLAV* approaches to relate schemas.

```sql
SELECT (basePrice * (1 + taxRate)) As Price FROM Products, Locations 
WHERE Products.saleLocID = Location.lid
```

```sql
# Obtain an entire tuple for items table of AGGREGATOR
SELECT title AS name, releaseDate AS releaseInfo, rating AS classification,  basePrice * (1 + taxRate) AS price FROM Movies, Products, Locations 
WHERE Movies.id = Products.mid AND Products.saleLocID = Locations.lid
```

- **Semantic Matches**: Relates *a set of elements* in *schema S* to *a set of elements* in *schema T*. **One-to-One**: ```Movies.title = Items.Name```; **Many-to-Many**: ```Items.price = Products.basePrice * (1 + Locations.taxRate)```.
- **Relationships**: start by creating semantic matches, elaborate them into mappings. 
- **Challenges**: semantic heterogeneity between schemas.
- Matchers -> Combiner -> Constraint Enforcer -> Match Selector. ```IN```: matches. ```Out```: mappings.

## 5.2 Schema Matching

#### 5.2.1. Matchers

- Input: S and T and other information. Output: sim matrix that assigns to *each element pair* of S and T a number in [0, 1], predict if the pair match. e.g. name = <name: 1, title: 0.2>; price = <basePrice: 0.8>.
- **Name-Based Matches**: Use string matching techniques and need pre-process.
- **Instance-Based Matches**: 1. **Recognizers**: use dict, regexes, or rules to recognize data values of attributes; 2. **Overlap Matches**: applies to attributes whose values are drawn from finite domain; 3. **Classifiers**: for each element $s_i$ in S, train classifier $C_i$ to recognize ins of $s_i$. Need pos & neg ins.

#### 5.2.2. Combining Match Predictions

- A combiner merges the sim martices output by different matchers into a single matrix. Can use Avg / Max / Min / Weighted Avg. 

#### 5.2.3. Enforcing Domain Integrity Constraints

- **Constraint Enforcer** eploits designer's domain knowledge to prune certain match combinations. Has hard and soft (more heuristic) cost.
- **A* Search**: guaranteed to find the optimal solution; computationally expensive.
- **Local Propagation**: faster, but performs only local optimizations.

#### 5.2.4. Match Selector

- Selects matches from the sim matrix. A simple strategy is thresholding / stable match.

#### 5.2.5. Reusing Previous Matches

- **Multi-Strategy Learning**: 1. Manually match $S_1$, …, $S_m$ into $G$; 2. Generalizes these matches to predict matches for $S_{m+1}, …, S_{n}$.

#### 5.2.6. Many-to-Many Mappings

- consider matches between combinations of columns. Employs text searcher, numeric searcher, and data searcher. Explore concatenations of all sizes.

- ```sql
  SELECT P.HrRate * W.hrs from PayRate P, WorksOn W WHERE P.Rank = W.ProjRank
  ```

# Distributed and Parallel Databases

- **Parallelism**: multiple CPUs, different computers wired together; **Parallel DBs**: racks of DBs on the same server

**Why we need parallel DBs**:

- **Types**: shared memories (CPU-Cache--BUS-Memory); shared disk (CPU-Cache-Mem—Disk); Shared Nothing (by network)

- **How to paralyze**: Pipelining (each job to its own machine), Partitioning (split data, process and merge)

- **Split data**: **Hash** (keep original key, not good with skewed data); **Range partitioning** (evenly distribute data); **randomly** (worst); **Workload Driven Partition** (best if size, activity or affinity known); **Round Robin** (distribute equally, 2nd worst).

- **Paralyze Join:** Same hash function (1-1 ID match, cheap), different hash functions (need to broadcast to all others, expensive); Known where the record is from A but not B (split B nodes and hash on B.ID, send data to A partition, slow); Neither A nor B hashed on ID (split both on ID, redistribute the data to do the join, expensive)

​            **Semi-Join**: send the data everywhere to look for a match

​            **Bloom join**: send bloom filter everywhere, they have false + and leads to more data than expected.

**CAP theorem**: for distributed DB only two out of Consistency (all nodes agree on the data), Availability (able to respond to requests), Partition Tolerance (some nodes cannot talk to each other, but you can still move forward)

**Primary**: one node is primary, others are secondary, read and write need to go through the primary node

**Multi-master**: read and write can arrive at any of the nodes

**Update**: Synchronous (primary wait for secondaries to finish); Asynchronous (confirms secondaries told).

# Warehousing and Columnar Databases

- **OLTP**: online transaction processing, r/w data, no analytics;  **OLAP**: analytical processing, data warehouse.
- **Column-oriented database**: read and process the columns that are necessary, columns have to all stay in the same order, inefficient for lookup or update.
- **Late Materialization**: get the data as late as possible, works on cols instead of rows when available.
- **Bit mapper**: bit vector as a map and apply it on the database, get relevant data fast.
- **Compression**: lossy & lossless, works better on column-oriented DBs, query performance and size trade-off.

![image-20190606125527341](/Users/ziyuye/Library/Application Support/typora-user-images/image-20190606125527341.png)

# GFS, BigTable, & MapReduce 

- **Google File System** (**GFS** or **GoogleFS**) is a [proprietary](https://en.wikipedia.org/wiki/Proprietary_software) [distributed file system](https://en.wikipedia.org/wiki/Distributed_file_system) developed by [Google](https://en.wikipedia.org/wiki/Google) to provide efficient, reliable access to data using large clusters of [commodity hardware](https://en.wikipedia.org/wiki/Commodity_hardware).
- GFS is enhanced for Google's core data storage and usage needs, which can generate enormous amounts of data that must be retained. Files are divided into fixed-size *chunks* of 64 [megabytes](https://en.wikipedia.org/wiki/Megabyte); files are usually appended to or read. It is also designed and optimized to run on Google's computing clusters, dense nodes which consist of cheap "commodity" computers, which means precautions must be taken against the high failure rate of individual nodes and the subsequent data loss. 
- A GFS cluster consists of multiple nodes, divided into two types: *Master* node and a large number of *Chunkservers*. Each file is divided into fixed-size chunks. Chunk servers store chunks. Each chunk is assigned a globally unique 64-bit label by the master node at creation, and logical mappings of files to constituent chunks are maintained. Each chunk is replicated several times throughout the network. At default, it is replicated three times, but this is configurable. Files which are in high demand may have a higher replication factor, while files for which the application client uses strict storage optimizations may be replicated less than three times - in order to cope with quick garbage cleaning policies.
- The Master server stores all the [metadata](https://en.wikipedia.org/wiki/Metadata) associated with the chunks, such as the tables mapping the 64-bit labels to chunk locations and the files they make up (mapping from files to chunks), the locations of the copies of the chunks, what processes are reading or writing to a particular chunk, or taking a "snapshot" of the chunk pursuant to replicate it. All this metadata is kept current by the Master server periodically receiving updates from each chunk server.
- Permissions for modifications are handled by a system of time-limited, expiring "leases", where the Master server grants permission to a process for a finite period of time during which no other process will be granted permission by the Master server to modify the chunk. The modifying chunkserver, which is always the primary chunk holder, then propagates the changes to the chunkservers with the backup copies. The changes are not saved until all chunkservers acknowledge, thus guaranteeing the completion and [atomicity](https://en.wikipedia.org/wiki/Atomicity_(database_systems)) of the operation.
- Programs access the chunks by first querying the Master server for the locations of the desired chunks; if the chunks are not being operated on, the Master replies with the locations, and the program then contacts and receives the data from the chunkserver directly. GFS is provided as a [userspace](https://en.wikipedia.org/wiki/Userspace) library.
- ![image-20190606124007779](/Users/ziyuye/Library/Application Support/typora-user-images/image-20190606124007779.png)

- **BigTable**: sparse, distributed, persistent, multi-dimensional -> sorted map.
- Bigtable is one of the prototypical examples of a [wide column store](https://en.wikipedia.org/wiki/Wide_column_store). It maps two arbitrary string values (row key and column key) and timestamp (hence three-dimensional mapping) into an associated arbitrary byte array. It is not a relational database and can be better defined as a **sparse, distributed multi-dimensional sorted map**.Bigtable is designed to scale into the [petabyte](https://en.wikipedia.org/wiki/Petabyte) range across "hundreds or thousands of machines, and to make it easy to add more machines to the system and automatically start taking advantage of those resources without any reconfiguration". For example, Google's copy of the web can be stored in a bigtable where the row key is a [domain-reversed URL](https://en.wikipedia.org/wiki/Reverse_domain_name_notation), and columns describe various properties of a web page, with one particular column holding the page itself. The page column can have several timestamped versions describing different copies of the web page timestamped by when they were fetched. Each cell of a bigtable can have zero or more timestamped versions of the data. Another function of the timestamp is to allow for both [versioning](https://en.wikipedia.org/wiki/Version_control) and [garbage collection](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)) of expired data.
