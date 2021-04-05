from ruamel import yaml

def generate_yaml_doc_ruamel(py_object,yaml_file):
	'''
	生成yaml 文件
	py_object：json or dict
	'''
    
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file, Dumper=yaml.RoundTripDumper)
    file.close()

def get_yaml_data_ruamel(yaml_file):
    '''
    读取yaml 文件
    '''
    file = open(yaml_file, 'r', encoding='utf-8')
    data = yaml.load(file.read(), Loader=yaml.Loader)
    file.close()
    return data
