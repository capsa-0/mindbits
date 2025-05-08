def calculate_compatibility(genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
    # Convertir genes a diccionario para acceso rápido por innovación
    genes1 = {gene.innovation: gene for gene in genome1.genes}
    genes2 = {gene.innovation: gene for gene in genome2.genes}
    
    # Conjuntos de claves
    keys1 = set(genes1.keys())
    keys2 = set(genes2.keys())

    # Genes coincidentes
    matching = keys1 & keys2
    weight_diff = 0.0
    for key in matching:
        weight_diff += abs(genes1[key].weight - genes2[key].weight)
    weight_diff = weight_diff / len(matching) if matching else 0

    # Disjuntos y excesivos
    max_innov1 = max(keys1, default=0)
    max_innov2 = max(keys2, default=0)
    max_shared = min(max_innov1, max_innov2)

    disjoint = [k for k in keys1 ^ keys2 if k <= max_shared]
    excess = len([k for k in keys1 ^ keys2 if k > max_shared])

    # N = longitud del genoma más largo (normalizador)
    N = max(len(genes1), len(genes2))
    if N < 20:  # se suele usar 1 si son pequeños
        N = 1

    return (c1 * excess + c2 * len(disjoint)) / N + c3 * weight_diff


def speciate(genomes, threshold=3.0):
    """Clasifica los genomas en especies usando un threshold de compatibilidad."""
    species = []  # lista de listas de tuplas (genome_id, genome_obj)
    
    for gid, genome in genomes:
        placed = False
        for especie in species:
            # Compara con el primer miembro de la especie
            if calculate_compatibility(especie[0][1], genome) < threshold:
                especie.append((gid, genome))
                placed = True
                break
        if not placed:
            species.append([(gid, genome)])
    
    # Asignar ID de especie a cada genome_id
    species_ids = {}
    for i, especie in enumerate(species):
        for gid, _ in especie:
            species_ids[gid] = i
    return species_ids
