from dijkstra.dijkstra import dijkstra, shortest_path

graph = {'a':{'b':1},
         'b': {'c': 2, 'b':5},
         'c':{'d':1},
         'd': {}}

dist, pred = dijkstra(graph, start='a')

print("This works")
