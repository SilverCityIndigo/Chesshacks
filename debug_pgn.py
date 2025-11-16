#!/usr/bin/env python
with open('src/resources/alphazero_games', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the first 'N' separator
idx = content.find(' 1-0\n')
if idx >= 0:
    snippet = content[idx:idx+100]
    print('After "1-0":')
    print(repr(snippet))

# Count [Event occurrences
count = content.count('[Event')
print(f'\nTotal [Event occurrences: {count}')
