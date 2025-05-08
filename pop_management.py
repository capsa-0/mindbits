import json
import torch



def guardarPoblacion(poblacion, nombre_archivo, nn_attribute, n_gen, repro_rate):
    print('Guardando población')
    def extraer_pesos_y_biases(red_neuronal):
        """Extrae pesos y biases de un modelo FeedForward de PyTorch"""
        state_dict = red_neuronal.state_dict()
        return {
            'weights': [state_dict[key].cpu().numpy().tolist() for key in state_dict if 'weight' in key],
            'biases': [state_dict[key].cpu().numpy().tolist() for key in state_dict if 'bias' in key]
        }
    
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f'{n_gen}\n')
        archivo.write(str(repro_rate)[1:-1]+'\n')

        for ind in poblacion:
            red_neuronal = getattr(ind, nn_attribute)
            color = getattr(ind, 'original_color')
            genoma_dict = extraer_pesos_y_biases(red_neuronal)
            genoma_dict['color'] = color

            json.dump(genoma_dict, archivo)
            archivo.write('\n')

def cargarPoblacion(archivo, ind_class, nn_class, input_size, arch):
    print('Cargando población')
    poblacion = []
    
    with open(archivo, 'r') as archivo:
        lineas = archivo.readlines()
        n_gen = int(lineas[0].strip())
        repro_rate = [float(x) for x in lineas[1].strip().split(',')]

        for linea in lineas[2:]:
            genoma_dict = json.loads(linea.strip())
            ind_color = genoma_dict['color']
            # Reconstruir state_dict con nombres correctos
            state_dict = {}
            
            # Asignar pesos y biases en orden
            for i, (w, b) in enumerate(zip(genoma_dict['weights'], genoma_dict['biases'])):
                state_dict[f'layers.{i}.weight'] = torch.tensor(w, dtype=torch.float32)
                state_dict[f'layers.{i}.bias'] = torch.tensor(b, dtype=torch.float32)
            
            # Crear red y cargar parámetros
            nn = nn_class(input_size, arch)
            nn.load_state_dict(state_dict)
            
            poblacion.append(ind_class(None, nn, color=ind_color))
    
    return poblacion, n_gen, repro_rate