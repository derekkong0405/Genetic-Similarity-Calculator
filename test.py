import numpy as np
import pandas as pd
import networkx as nx
import pronto
import random

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import jax.numpy as jnp

import jax

from scipy.sparse import lil_matrix


# Assign an organ system
# First load HPO OBO
ontology = pronto.Ontology("hp.obo")
# Instantiate a graph
hpo_graph = nx.DiGraph()
# Add nodes and edges from the OBO file
for term in ontology.terms():
    # print(term)
    hpo_graph.add_node(term.id, name=term.name)
    for parent in term.superclasses(distance=1):
        hpo_graph.add_edge(parent.id, term.id)

hpo_graph.remove_edges_from([(n, n) for n in hpo_graph.nodes()])

# print(hpo_graph.nodes)

# Load the genes_to_phenotype file
df = pd.read_csv("genes_to_phenotype.txt", sep="\t")

# print(df.head())

# Encode gene symbols and HPO IDs as integer indices
gene_encoder = LabelEncoder()

gene_indices = gene_encoder.fit_transform(df["gene_symbol"])
# phenotype_indices = phenotype_encoder.fit_transform(df["hpo_id"])
# print(df["hpo_id"])

# print(phenotype_indices)

all_nodes = list(hpo_graph.nodes())

phenotype_encoder = LabelEncoder()
phenotype_encoder.classes_ = np.array(all_nodes)
phenotype_encoder.fit(list(hpo_graph.nodes()))
# print(len(list(hpo_graph.nodes())))

# Now transform the phenotype IDs from the CSV into indices aligned with the full ontology
phenotype_indices = phenotype_encoder.transform(df["hpo_id"])


# Each row of the matrix is (gene, phenotype) with value 1
data = [1] * len(df)
row_indices = gene_indices
col_indices = phenotype_indices

# Create sparse matrix
gene_matrix = csr_matrix(
    (data, (row_indices, col_indices)),
    shape=(len(gene_encoder.classes_), len(phenotype_encoder.classes_))
)

gene_matrix.data = np.ones_like(gene_matrix.data)



# Optional: inspect shape
# print(hpo_graph.nodes())
# print(f"Matrix shape: {adj_matrix.shape}")  # (num_genes, num_phenotypes)


# Get all nodes in a fixed order
all_nodes = list(hpo_graph.nodes())
node_to_index = {node: i for i, node in enumerate(all_nodes)}

# Initialize a sparse square matrix
N = len(all_nodes)
ancestor_matrix = lil_matrix((N, N), dtype=np.uint8)

# Fill matrix: for each node, mark its ancestors
for child in all_nodes:
    child_idx = node_to_index[child]
    ancestors = nx.ancestors(hpo_graph, child)  # all transitive ancestors
    for ancestor in ancestors:
        ancestor_idx = node_to_index[ancestor]
        ancestor_matrix[child_idx, ancestor_idx] = 1

# Optionally convert to CSR for efficient ops
ancestor_matrix = ancestor_matrix.tocsr()

adj_matrix_propagated = gene_matrix @ ancestor_matrix

adj_matrix_propagated.data = np.ones_like(adj_matrix_propagated.data)

num_occurrence = total_sum = adj_matrix_propagated.sum()

# 1. Sum the propagated matrix over columns to get phenotype counts
phenotype_counts = np.array(adj_matrix_propagated.sum(axis=0)).flatten()  # shape (num_phenotypes,)

# 2. Total number of annotations (or occurrences)
num_occurrence = adj_matrix_propagated.sum()

def compute_ic(phenotype_counts, num):
    # print("Max phenotype_counts:", np.max(phenotype_counts))
    # print("num_genes:", num_genes)
    # print("Max p (phenotype_counts / num_genes):", np.max(phenotype_counts / num_genes))
    p = phenotype_counts / num
    epsilon = 1e-10
    ic = -np.log(p + epsilon)
    return ic

# Suppose phenotype_counts_jax and num_genes_jax are jax arrays
icVector = jnp.array(compute_ic(phenotype_counts, num_occurrence))

# print(icVector)

@jax.jit
def simple_gene_resnik(gene_vec_A, gene_vec_B, ic_vector):
    shared = gene_vec_A & gene_vec_B
    # mask ic_vector where shared=0 with -inf (lowest possible)
    masked_ic = jnp.where(shared, ic_vector, -jnp.inf)
    max_ic = jnp.max(masked_ic)
    # if max_ic is -inf, means no shared phenotypes, return 0.0 else max_ic
    return jnp.where(max_ic == -jnp.inf, 0.0, max_ic)

@jax.jit
def sum_gene_resnik(gene_vec_A, gene_vec_B, ic_vector):
    shared = gene_vec_A & gene_vec_B
    # select ic_vector values where shared is True, else 0
    shared_ic = jnp.where(shared, ic_vector, 0.0)
    total_ic = jnp.sum(shared_ic)
    return total_ic

@jax.jit
def simgic_alpha(gene_vec_A, gene_vec_B, ic_vector, alpha=1.2):
    # print("IC vector min:", jnp.min(ic_vector))
    # print("Any negative?", jnp.any(ic_vector < 0))
    # print("Any NaN in IC vector?", jnp.any(jnp.isnan(ic_vector)))
    # print("Any inf in IC vector?", jnp.any(jnp.isinf(ic_vector)))
    shared = gene_vec_A & gene_vec_B       # A ∩ B (bitwise AND)
    union = gene_vec_A | gene_vec_B        # A ∪ B (bitwise OR)

    # Raise IC to the power alpha
    ic_alpha = jnp.power(ic_vector, alpha)

    numerator = jnp.sum(jnp.where(shared, ic_alpha, 0.0))
    denominator = jnp.sum(jnp.where(union, ic_alpha, 0.0))

    # Handle case where denominator == 0 (no terms in union)
    return jnp.where(denominator == 0, 0.0, numerator / denominator)


def get_gene_vector(gene_symbol, gene_encoder, adj_matrix):
    # Get gene index from symbol
    gene_idx = gene_encoder.transform([gene_symbol])[0]
    # Extract binary phenotype vector (dense)
    gene_vec = adj_matrix[gene_idx].toarray().flatten()
    # Convert to jnp array for JAX functions
    return jnp.array(gene_vec)

# Example gene symbols
gene_A = "YIF1B"
gene_B = "PGAP1"

# Get phenotype vectors
vec_A = get_gene_vector(gene_A, gene_encoder, adj_matrix_propagated)
vec_B = get_gene_vector(gene_B, gene_encoder, adj_matrix_propagated)

# Calculate Resnik similarity
sim = simgic_alpha(vec_A, vec_B, icVector)

print(f"SimGIC similarity between {gene_A} and {gene_B}: {sim}")

def find_most_similar_gene(query_gene, gene_list, gene_encoder, adj_matrix, icVector):
    query_vec = get_gene_vector(query_gene, gene_encoder, adj_matrix)

    def compute_sim(gene_symbol):
        gene_vec = get_gene_vector(gene_symbol, gene_encoder, adj_matrix)
        return simgic_alpha(query_vec, gene_vec, icVector)

    sims = jnp.array([compute_sim(g) for g in gene_list])
    max_idx = jnp.argmax(sims)
    return gene_list[int(max_idx)], float(sims[max_idx])

# Example usage:
query = "CFAP58"
candidate_genes = ["DCAF17", "TCTN2", "TBL1XR1", "HPS6"]

most_similar_gene, similarity_score = find_most_similar_gene(query, candidate_genes, gene_encoder, adj_matrix_propagated, icVector)
print(f"Most similar gene to {query} is {most_similar_gene} with SimGIC similarity {similarity_score:.4f}")


# # Optional: print example nodes with layers
# for node in list(hpo_graph.nodes)[:10]:
#     print(f"{node}: {hpo_graph.nodes[node]['name']} (Layer {hpo_graph.nodes[node].get('layer')})")



# View basic information about the graph
# print("Number of nodes:", hpo_graph.number_of_nodes())
# print("Number of edges:", hpo_graph.number_of_edges())

# # View nodes and edges
# print("Nodes:", list(hpo_graph.nodes()))
# print("Edges:", list(hpo_graph.edges()))

# hpo_rev_graph = hpo_graph.reverse()

# print("Number of nodes:", hpo_graph.number_of_nodes())
# print("Number of edges:", hpo_graph.number_of_edges())

# # Print random edge from the original graph
# original_edge = random.choice(list(hpo_graph.edges()))
# print("Original edge:", original_edge)
#
# # Check if the reversed edge exists in the reversed graph
# reversed_edge = (original_edge[1], original_edge[0])  # Reverse the edge direction
# print("Reversed edge:", reversed_edge)
#
# # Check if the reversed edge is in the reversed graph
# if reversed_edge in hpo_rev_graph.edges():
#     print(f"Edge {reversed_edge} exists in the reversed graph!")
# else:
#     print(f"Edge {reversed_edge} does not exist in the reversed graph.")

# Check for self-loops in the graph
# self_loops = [edge for edge in hpo_graph.edges() if edge[0] == edge[1]]
#
# # Print the self-loops
# if self_loops:
#     print("Self-loops found:", self_loops)
# else:
#     print("No self-loops found.")

# def detect_cycle_dfs(graph, node, visited, rec_stack):
#     # Mark the current node as visiting
#     visited[node] = True
#     rec_stack[node] = True
#
#     # Explore all the neighbors (parents in this case) of the current node
#     for neighbor in graph.neighbors(node):
#         if not visited[neighbor]:  # If the neighbor hasn't been visited
#             if detect_cycle_dfs(graph, neighbor, visited, rec_stack):
#                 return True
#         elif rec_stack[neighbor]:  # If the neighbor is in the recursion stack
#             return True
#
#     # After visiting all neighbors, remove the node from the recursion stack
#     rec_stack[node] = False
#     return False
#
# # Function to check for cycles in the HPO graph
# def detect_cycles_in_hpo_graph(hpo_graph):
#     # Initialize the visited and recursion stack dictionaries
#     visited = {node: False for node in hpo_graph.nodes()}
#     rec_stack = {node: False for node in hpo_graph.nodes()}
#
#     # Check for cycles in each node
#     for node in hpo_graph.nodes():
#         if not visited[node]:  # If the node has not been visited
#             if detect_cycle_dfs(hpo_graph, node, visited, rec_stack):
#                 print(f"Cycle detected starting at node {node}")
#                 return True
#     return False
#
# # Assuming you have already created the hpo_graph from the OBO terms and edges
# # Run cycle detection on your graph
# if detect_cycles_in_hpo_graph(hpo_graph):
#     print("The HPO graph contains cycles.")
# else:
#     print("The HPO graph does not contain cycles.")


# # 2. Count annotated genes per phenotype (column sum)
# # adj_matrix is scipy sparse, so sum over axis=0 gives count per phenotype
# phenotype_counts = np.array(adj_matrix.sum(axis=0)).flatten()  # shape (num_phenotypes,)
#
# # Map phenotype indices back to HPO IDs
# phenotype_index_to_id = dict(enumerate(phenotype_encoder.classes_))
#
# # Build dictionary of raw counts {hpo_id: count}
# raw_counts_dict = {
#     phenotype_index_to_id[i]: phenotype_counts[i]
#     for i in range(len(phenotype_counts))
# }
#
# def propagate_counts(G, raw_counts_dict):
#     # Start with a default of 0 for all nodes
#
#     for node in reversed(list(nx.topological_sort(G))):
#         for parent in G.predecessors(node):
#             raw_counts_dict[parent] += raw_counts_dict[node]
#
#     genes = sum(raw_counts_dict.values())
#
#     return raw_counts_dict, genes
#
# phenotype_counts_propagated_dict, num_occurrence = propagate_counts(hpo_graph, raw_counts_dict)
#
# # Back to array aligned with phenotype_encoder.classes_
# phenotype_counts_propagated = np.array([
#     phenotype_counts_propagated_dict.get(hpo_id, 0.0)
#     for hpo_id in phenotype_encoder.classes_
# ])
