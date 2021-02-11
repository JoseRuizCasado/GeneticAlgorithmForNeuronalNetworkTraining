import random

example = []
for i in range(10):
    example.append({
        'model': f'modelo {i}',
        'fitness': random.random()
    })
print(f'List {example}')
print(example.sort(key=lambda x: x['fitness']))


# Initializing list of dictionaries
lis = [{ "name" : "Nandini", "age" : 20},
       { "name" : "Manjeet", "age" : 20 },
{ "name" : "Nikhil" , "age" : 19 }]

print(sorted(example, key=lambda i: i['fitness']))


