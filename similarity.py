
import jax.numpy as jnp

import jax


from setup import icVector
#from setup_jaxSparse import icVector



#simply returns maximum IC of their shared phenotypes (including ancestors on hierarchy)
@jax.jit
def simple_gene_resnik(gene_vec_A, gene_vec_B, ic_vector):
    shared = gene_vec_A & gene_vec_B
    # mask ic_vector where shared=0 with -inf (lowest possible)
    masked_ic = jnp.where(shared, ic_vector, -jnp.inf)
    max_ic = jnp.max(masked_ic)
    # if max_ic is -inf, means no shared phenotypes, return 0.0 else max_ic
    return jnp.where(max_ic == -jnp.inf, 0.0, max_ic)

#sums up all their combined ICs of all shared phenotypes
@jax.jit
def sum_gene_resnik(gene_vec_A, gene_vec_B, ic_vector):
    shared = gene_vec_A & gene_vec_B
    # select ic_vector values where shared is True, else 0
    shared_ic = jnp.where(shared, ic_vector, 0.0)
    total_ic = jnp.sum(shared_ic)
    return total_ic

#now also takes into account how many other phenotypes they don't share and minimizes for not shared phenotypes
@jax.jit
def simgic_alpha(gene_vec_A, gene_vec_B, ic_vector, alpha=1.2):
    shared = gene_vec_A & gene_vec_B       # A ∩ B (bitwise AND)
    union = gene_vec_A | gene_vec_B        # A ∪ B (bitwise OR)

    # Raise IC to the power alpha, emphasizes "rare" ICs even further
    ic_alpha = jnp.power(ic_vector, alpha)

    numerator = jnp.sum(jnp.where(shared, ic_alpha, 0.0))
    denominator = jnp.sum(jnp.where(union, ic_alpha, 0.0))

    # Handle case where denominator == 0 (no terms in union)
    return jnp.where(denominator == 0, 0.0, numerator / denominator)

# Vectorized version: given a single gene_vec_A and batch of gene_vec_Bs
@jax.jit
def batch_simgic_alpha(query_vec, batch_gene_vecs, ic_vector = icVector, alpha=1.2):
    # Vectorize simgic_alpha over second argument (axis 0 of batch_gene_vecs)
    return jax.vmap(lambda gene_vec_B: simgic_alpha(query_vec, gene_vec_B, ic_vector, alpha))(batch_gene_vecs)



#debugging simgic, NO JAX PRINTS
# def simgic_alpha_with_report(gene_vec_A, gene_vec_B, ic_vector, all_nodes, alpha=1.2):
#     shared = jnp.logical_and(gene_vec_A, gene_vec_B)
#     union = jnp.logical_or(gene_vec_A, gene_vec_B)
#     difference = jnp.logical_xor(gene_vec_A, gene_vec_B)
#
#     ic_alpha = jnp.power(ic_vector, alpha)
#
#     numerator = jnp.sum(jnp.where(shared, ic_alpha, 0.0))
#     denominator = jnp.sum(jnp.where(union, ic_alpha, 0.0))
#
#     sim_score = jnp.where(denominator == 0, 0.0, numerator / denominator)
#
#     # 5. Report shared terms and ICs (outside JAX)
#     shared_indices = jnp.where(shared)[0]
#     shared_indices_list = shared_indices.tolist()
#     diff_list = jnp.where(difference)[0].tolist()
#     diff_ids = [all_nodes[i] for i in diff_list]
#     shared_ids = [all_nodes[i] for i in shared_indices_list]
#     shared_ic_values = [float(ic_vector[i]) for i in shared_indices_list]
#     shared_ic_alpha_values = [float(ic_vector[i] ** alpha) for i in shared_indices_list]
#
#     print("Shared term IDs:", shared_ids)
#     print("Not Shared:", diff_ids)
#     print(f"IC^{alpha} values:", shared_ic_alpha_values)
#     print(numerator)
#     print(jnp.sum(jnp.where(difference, ic_alpha, 0.0)))
#     print(denominator)
#
#     return float(sim_score)





