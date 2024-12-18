import sys
with open(sys.argv[1]) as inp:
    with open(sys.argv[2], 'w') as out:
        g = [list(map(eval, line.split())) for line in inp]
        g.sort()
        for g in g:
            print(*g, sep='\t', file=out)


