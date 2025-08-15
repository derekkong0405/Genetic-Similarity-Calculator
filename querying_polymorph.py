import numpy as np
import pandas as pd
import networkx as nx
import pronto

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import jax.numpy as jnp

import jax

from jax import export
from jax import ShapeDtypeStruct

from jax.core import raise_to_shaped

from scipy.sparse import lil_matrix

from setup import adj_matrix_propagated, gene_encoder, N
from similarity import batch_simgic_alpha


#make sure to only query the same amount each time otherwise jax has to recompile (bc fundamentally a 10 row matrix
#operation function is different from an 11 row, so it has to reoptimize and retrace
def extract_gene_vectors(gene_names, gene_encoder = gene_encoder, adj_matrix = adj_matrix_propagated):
    gene_vectors = []

    for gene_name in gene_names:
        try:
            # Transform gene name to index
            index = gene_encoder.transform([gene_name])[0]
            # Extract the gene vector (dense from sparse matrix)
            gene_vector = adj_matrix[index].toarray().flatten()
            # Append the vector
            gene_vectors.append(gene_vector)
        except ValueError:
            # If gene is not found, append a zero vector
            print(f"Gene '{gene_name}' not found, adding a zero vector.")
            gene_vectors.append(np.zeros(adj_matrix.shape[1], dtype=np.uint8))  # Zero vector

    # Convert list of vectors to JAX array
    return jnp.array(gene_vectors, dtype=jnp.uint8)

# def extract_gene_vectors(gene_names, gene_encoder = gene_encoder, adj_matrix = adj_matrix_propagated):
#     # Transform gene names to indices
#     indices = gene_encoder.transform(gene_names)
#     # Extract dense vectors from sparse matrix
#     vectors = jnp.array(adj_matrix[indices].toarray(), dtype=jnp.uint8)
#     print(vectors)
#     return vectors


# Assuming 'N' is a global or passed configuration for the dimension.
# Let's use a concrete number for this example.
DIMENSION = 5 # Example dimension, replace with your actual HPO terms count (e.g., 1234)

# Define symbolic dimension "n" for the number of candidate genes
N_SYMBOLIC = export.symbolic_shape("n")

# Define polymorphic shapes
POLY_SHAPE_CANDIDATES = (N_SYMBOLIC, DIMENSION)
POLY_SHAPE_QUERY = (DIMENSION,)

@jax.jit
def jitted_similarity_core(query_vec: jax.Array, candidate_vecs: jax.Array):
    # # Compute similarity scores for all candidate genes
    # jax.debug.print("genes in position: {}", candidate_vecs)
    # jax.debug.print("zero: {}", adj_matrix_propagated[gene_encoder.transform(["B3GLCT"])[0]].toarray().flatten())
    # jax.debug.print("one: {}", adj_matrix_propagated[gene_encoder.transform(["TSC2"])[0]].toarray().flatten())

    sims = batch_simgic_alpha(query_vec, candidate_vecs)
    # jax.debug.print("Similarity scores: {}", sims)

    # Create a JAX array of indices and similarity scores
    indices = jnp.arange(len(sims))
    # jax.debug.print("gene indices: {}", indices)

    # Stack indices with similarity scores (column stack)
    indexed_sims = jnp.column_stack((indices, sims))
    # jax.debug.print("similarity scores mapped: {}", indexed_sims)

    # Sort by similarity score in descending order
    sorted_indices = jnp.argsort(-indexed_sims[:, 1])  # Sort by second column (the score)

    # Ensure sorted_indices are integers (casting to int32)
    sorted_indices = jnp.astype(sorted_indices, jnp.int32)

    # Now we can safely index into the array
    sorted_indexed_sims = indexed_sims[sorted_indices]
    # jax.debug.print("sorted order: {}", sorted_indexed_sims)

    # Extract sorted scores and gene indices
    sorted_scores = sorted_indexed_sims[:, 1]
    # jax.debug.print("sorted scores:{}", sorted_scores)
    sorted_gene_indices = sorted_indexed_sims[:, 0]
    # jax.debug.print("sorted indices:{}", sorted_gene_indices)

    return sorted_gene_indices, sorted_scores


# --- Compile the JAX function (Only once at startup) ---
COMPILED_SIM_FN = export.export(jitted_similarity_core)(
    ShapeDtypeStruct(shape=POLY_SHAPE_QUERY, dtype=jnp.float32),
    ShapeDtypeStruct(shape=POLY_SHAPE_CANDIDATES, dtype=jnp.float32))

def find_most_similar_genes_dynamic(query_gene, candidate_genes, dimension = N):

    # Get the single query vector
    query_vec = extract_gene_vectors([query_gene])[0]
    # Get all candidate vectors
    candidate_vecs = extract_gene_vectors(candidate_genes)

    sorted_idx, scores = COMPILED_SIM_FN(query_vec, candidate_vecs)

    # Debug print for sorted_idx and scores
    # jax.debug.print("sorted_idx: {}", sorted_idx)
    # jax.debug.print("scores: {}", scores)

    ordered_idx = jnp.arange(len(candidate_genes))

    # Sort genes and scores using sorted_idx
    sorted_genes = [candidate_genes[int(idx)] for idx in sorted_idx]
    sorted_scores = [float(scores[int(idx)]) for idx in ordered_idx]

    # Debug print for sorted genes and sorted scores
    # jax.debug.print("sorted_genes: {}", sorted_genes)
    # jax.debug.print("sorted_scores: {}", sorted_scores)

    # Return the result
    result = list(zip(sorted_genes, sorted_scores))

    return result



gene_names = ["A", "TSC2", "BRCA1", "FBLN5"]
res = find_most_similar_genes_dynamic("TSC1",gene_names)
print(res)



# # Example gene symbols
# gene_A = "TSC1"
# gene_B = "TSC2"
#
# # Get phenotype vectors
# vec_A = get_gene_vector(gene_A, gene_encoder, adj_matrix_propagated)
# vec_B = get_gene_vector(gene_B, gene_encoder, adj_matrix_propagated)
#
# # Calculate similarity
# sim = simgic_alpha(vec_A, vec_B, icVector,all_nodes)
#
# print(f"SimGIC similarity between {gene_A} and {gene_B}: {sim}")

# #Example query:
# query = "CFAP58"
# #List of Candidates
# candidate_genes = ["DCAF17", "TCTN2", "TBL1XR1", "HPS6"]
# #calculation
# most_similar_gene, similarity_score = find_most_similar_gene(query, candidate_genes, gene_encoder, adj_matrix_propagated, icVector)
# print(f"Most similar gene to {query} is {most_similar_gene} with SimGIC similarity {similarity_score:.4f}")