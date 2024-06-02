i = 0
with Chem.MultithreadedSDMolSupplier('data/5ht3ligs.sdf') as sdSupl:
  for mol in sdSupl:
    if mol is not None:
      i += 1

