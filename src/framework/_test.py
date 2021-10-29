import json
source = {}
target = {}
file_source = open('./data/IKAMI/D_W_15K_V2/source.txt', 'r')
lines_source = file_source.readlines()
for line in lines_source:
    l = line.strip().split("\t")
    source[l[0]] = l[1]

file_target = open('./data/IKAMI/D_W_15K_V2/target.txt', 'r')
lines_target = file_target.readlines()
for line in lines_target:
    l = line.strip().split("\t")
    target[l[0]] = l[1]

file_rel_triples_1 = open('./data/IKAMI/D_W_15K_V2/rel_triples_1', 'r')
lines_triples_1 = file_rel_triples_1.readlines()
with open('./data/IKAMI/D_W_15K_V2/rel_triples_1.txt', 'w') as file:
    for line in lines_triples_1:
        l = line.strip().split("\t")
        file.write(source[l[0]]+ '\t'+ source[l[2]] + '\n')


file_rel_triples_2 = open('./data/IKAMI/D_W_15K_V2/rel_triples_2', 'r')
lines_triples_2 = file_rel_triples_2.readlines()
with open('./data/IKAMI/D_W_15K_V2/rel_triples_2.txt', 'w') as file:
    for line in lines_triples_2:
        l = line.strip().split("\t")
        file.write(target[l[0]]+ '\t'+ target[l[2]] + '\n')