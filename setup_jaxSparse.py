import numpy as np
import pandas as pd
import networkx as nx
import pronto

from sklearn.preprocessing import LabelEncoder
import jax.numpy as jnp

import jax #why isn't it using this import

import jax.experimental.sparse

from scipy.sparse import lil_matrix

#load obo (basically the hierarchy)
ontology = pronto.Ontology("hp.obo")
#make it a graph
hpo_graph = nx.DiGraph()
# Add nodes and edges from the OBO file
for term in ontology.terms():
    hpo_graph.add_node(term.id, name=term.name)
    #only edges for direct parents
    #need self loops for multiplication
    hpo_graph.add_edge(term, term)
    for parent in term.superclasses(distance=1):
        hpo_graph.add_edge(parent.id, term.id)

#remove self loops
#hpo_graph.remove_edges_from([(n, n) for n in hpo_graph.nodes()])

# Load the genes_to_phenotype file
df = pd.read_csv("genes_to_phenotype.txt", sep="\t")



# Fixed order of phenotypes (HPO nodes)
all_nodes = list(hpo_graph.nodes())
node_to_index = {node: i for i, node in enumerate(all_nodes)}

# Encode phenotypes manually to avoid LabelEncoder sorting (need same phenotype ordering for matrix x)
phenotype_indices = df["hpo_id"].map(node_to_index).values

# Encode gene symbols (all genes should be present in df)
gene_encoder = LabelEncoder()
gene_indices = gene_encoder.fit_transform(df["gene_symbol"])

# Create sparse matrix (genes x phenotypes)
# Convert to JAX arrays
data = jnp.ones(len(df), dtype=jnp.uint8)
row_indices = jnp.array(gene_indices)
col_indices = jnp.array(phenotype_indices)

# Total number of rows = number of genes
n_rows = len(gene_encoder.classes_)
n_cols = len(all_nodes)

# Step 1: Count non-zeros per row
counts = jnp.bincount(row_indices, length=n_rows)

# Step 2: Build indptr (CSR-style row pointers)
indptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])

# Step 3: Build the BCSR matrix
gene_matrix = jax.experimental.sparse.BCSR((data, col_indices, indptr), shape=(n_rows, n_cols))

# Create ancestor matrix (phenotype -> its ancestors)
N = len(all_nodes)
ancestor_matrix = lil_matrix((N, N), dtype=np.uint8)

for child in all_nodes:
    child_idx = node_to_index[child]
    #self is ancestor here
    ancestor_matrix[child_idx, child_idx] = 1
    for ancestor in nx.ancestors(hpo_graph, child):
        ancestor_idx = node_to_index[ancestor]
        ancestor_matrix[child_idx, ancestor_idx] = 1

ancestor_matrix = jax.experimental.sparse.BCSR.from_scipy_sparse(ancestor_matrix)

# Propagate: (genes x phenotypes) @ (phenotypes x phenotypes) = (genes x phenotypes)
#no jax bc one time thing
#nvm want to use jax but unsure of if experimental sparse is worth it
@jax.jit
def propagate_adj_matrix(gene_matrix, ancestor_matrix):
    # Perform the matrix multiplication
    ancestor_matrix = jax.experimental.sparse.BCSR.todense(ancestor_matrix)
    adj_matrix_propagated = gene_matrix @ ancestor_matrix

    # Update the 'data' attribute to ones
    adj_matrix_propagated = jnp.ones_like(adj_matrix_propagated)

    return adj_matrix_propagated

adj_matrix_propagated = propagate_adj_matrix(gene_matrix, ancestor_matrix)






# 1. Sum the propagated matrix over columns to get phenotype counts
phenotype_counts = jnp.array(adj_matrix_propagated.sum(axis=0)).flatten()  # shape (num_phenotypes,)

# 2. Total number of annotations (or occurrences)
num_occurrence = jax.jit(lambda: jnp.sum(phenotype_counts))()

#don't use jax because only computed once
#nvm if we have a tpu
@jax.jit
def compute_ic(phenotype_counts, num):
    p = phenotype_counts / num
    #can't log 0 so protect agains that
    epsilon = 1e-12
    ic = -jnp.log(p + epsilon)
    return ic

# convert to jax array because might have to use in numerous computations later down line
icVector = compute_ic(phenotype_counts, num_occurrence)




# def validate_gene_annotation(
#         gene_name: str,
#         gene_encoder,
#         gene_matrix,
#         ancestor_matrix,
#         adj_matrix_propagated,
#         all_nodes,
#         node_to_index,
#         verbose=True
# ):
#     """
#     Validate the annotation propagation for a given gene.
#
#     Parameters:
#     - gene_name: str, gene symbol to validate
#     - gene_encoder: LabelEncoder fitted on genes
#     - gene_matrix: sparse matrix (genes x phenotypes) original direct annotations
#     - ancestor_matrix: sparse matrix (phenotypes x phenotypes) child->ancestors
#     - adj_matrix_propagated: sparse matrix (genes x phenotypes) propagated annotations
#     - all_nodes: list of phenotype IDs in order of matrix columns
#     - node_to_index: dict phenotype ID -> matrix column index
#     - verbose: whether to print details
#
#     Returns:
#     - dict with keys 'direct_terms', 'ancestor_terms', 'propagated_terms', 'issues'
#     """
#
#     issues = []
#     try:
#         gene_idx = gene_encoder.transform([gene_name])[0]
#     except ValueError:
#         raise ValueError(f"Gene '{gene_name}' not found in gene_encoder classes")
#
#     # Original direct annotations (phenotype indices)
#     direct_indices = gene_matrix[gene_idx].nonzero()[1]
#     direct_terms = {all_nodes[i] for i in direct_indices}
#
#     # Ancestors of all direct phenotypes using the DAG
#     ancestor_terms = set()
#     for hpo_id in direct_terms:
#         ancestor_terms.update(nx.ancestors(hpo_graph, hpo_id))
#
#     # Combined expected propagated terms = direct + ancestors
#     expected_propagated_terms = direct_terms.union(ancestor_terms)
#
#     # Actual propagated annotations from combined matrix
#     propagated_indices = adj_matrix_propagated[gene_idx].nonzero()[1]
#     propagated_terms = {all_nodes[i] for i in propagated_indices}
#
#     # Validation checks
#
#     # 1) Propagated terms should be superset of direct terms
#     if not direct_terms.issubset(propagated_terms):
#         issues.append(f"Propagated terms missing some direct terms: {direct_terms - propagated_terms}")
#
#     # 2) Propagated terms should include all ancestors of direct terms
#     if not ancestor_terms.issubset(propagated_terms):
#         issues.append(f"Propagated terms missing some ancestors: {ancestor_terms - propagated_terms}")
#
#     # 3) Propagated terms should not include any extra terms beyond direct + ancestors
#     extras = propagated_terms - expected_propagated_terms
#     if extras:
#         issues.append(f"Propagated terms include unexpected phenotypes: {extras}")
#
#     if verbose:
#         print(f"Gene: {gene_name}")
#         print(f"Direct annotated phenotypes ({len(direct_terms)}): {sorted(direct_terms)}")
#         print(f"Ancestor phenotypes ({len(ancestor_terms)}): {sorted(ancestor_terms)}")
#         print(f"Propagated phenotypes ({len(propagated_terms)}): {sorted(propagated_terms)}")
#         if issues:
#             print("Issues found:")
#             for issue in issues:
#                 print(f"  - {issue}")
#         else:
#             print("No issues found: propagation matches expected annotations.")
#
#     return {
#         "direct_terms": direct_terms,
#         "ancestor_terms": ancestor_terms,
#         "propagated_terms": propagated_terms,
#         "issues": issues
#     }

# result = validate_gene_annotation(
#     "B3GLCT",
#     gene_encoder,
#     gene_matrix,
#     ancestor_matrix,
#     adj_matrix_propagated,
#     all_nodes,
#     node_to_index,
#     verbose=True
# )
