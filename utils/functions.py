import os
def get_folder_size(path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Evitar errores con enlaces simbólicos rotos
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size