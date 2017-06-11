---
layout: post
title: Parsing Large SDF files with Python
---
As we would do with any problem of significant size, the first step in solving it is to break the problem down into lots of 'mini' problems.

My research project this summer at Ames National Lab has, thus far, proven to be no different!

While getting [TensorFlow](https://www.tensorflow.org/) to accurately predict the bonding of atoms in a molecule given a set of vectors detailing the position of the atoms in a 3D space will be difficult in and of itself, we first need to figure out how to get our training data to build our model!

## Getting the SDF Training File

Fortunately, the scientific community tends to be very generous and has made large datasets (really really large!) of molecules with exactly the data we need publicly available via FTP on [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

The specific directory of SDF files we need is:
```
ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF/
```

There are quite a few files in here and they are several million lines each so let's just grab the first one for now. If and when we need more training data later on we can grab more at that time. The script we are going to use is general enough where it can be ran against any file in this directory.

I'm on a mac so I'll be using curl but you could also use wget to do the same if you so choose.

```
curl -o train.sdf.gz ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound_3D/01_conf_per_cmpd/SDF/00000001_00025000.sdf.gz
```
Next we need to extract our file

```
gzip -d train.sdf.gz
```

## Parsing the SDF Training File
Now let's open up our file and have a look at the contents. I recommend opening it in vim. Vim is generally the fastest tool for opening up really big files. You can also use something else but the application might freeze for a moment while loading up the file since it is over 4 million lines!

```
1
  -OEChem-06151611163D

 31 30  0     1  0  0  0  0  0999 V2000
    0.3387    0.9262    0.4600 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.4786   -1.7069   -0.3119 O   0  5  0  0  0  0  0  0  0  0  0  0
    1.8428   -1.4073    1.2523 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.4166    2.5213   -1.2091 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2359   -0.7251    0.0270 N   0  3  0  0  0  0  0  0  0  0  0  0
   -0.7783   -1.1579    0.0914 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1368   -0.0961   -0.5161 C   0  0  2  0  0  0  0  0  0  0  0  0
   -3.1119   -1.7972    0.6590 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4103    0.5837    0.7840 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6433   -0.5289   -1.4260 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4879   -0.6438   -0.9795 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.3478   -1.3163    0.1002 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4627    2.1935   -0.0312 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6678    3.1549    1.1001 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7073   -2.1051   -0.4563 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5669   -1.3392    1.1503 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3089    0.3239   -1.4193 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.9705   -2.7295    0.1044 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.8083   -1.9210    1.7028 H   0  0  0  0  0  0  0  0  0  0  0  0
   -4.1563   -1.4762    0.6031 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.0398    1.4170    0.1863 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.4837    0.7378    0.9384 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9129    0.5071    1.7551 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.2450    0.4089   -1.8190 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.3000   -1.3879   -2.0100 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7365   -0.4723   -1.4630 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.3299   -1.3744   -1.7823 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0900    0.1756   -1.3923 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1953    3.1280    1.7699 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.7681    4.1684    0.7012 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.5832    2.9010    1.6404 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  7  1  0  0  0  0
  1 13  1  0  0  0  0
  2 12  1  0  0  0  0
  3 12  2  0  0  0  0
  4 13  2  0  0  0  0
  5  6  1  0  0  0  0
  5  8  1  0  0  0  0
  5  9  1  0  0  0  0
  5 10  1  0  0  0  0
  6  7  1  0  0  0  0
  6 15  1  0  0  0  0
  6 16  1  0  0  0  0
  7 11  1  0  0  0  0
  7 17  1  0  0  0  0
  8 18  1  0  0  0  0
  8 19  1  0  0  0  0
  8 20  1  0  0  0  0
  9 21  1  0  0  0  0
  9 22  1  0  0  0  0
  9 23  1  0  0  0  0
 10 24  1  0  0  0  0
 10 25  1  0  0  0  0
 10 26  1  0  0  0  0
 11 12  1  0  0  0  0
 11 27  1  0  0  0  0
 11 28  1  0  0  0  0
 13 14  1  0  0  0  0
 14 29  1  0  0  0  0
 14 30  1  0  0  0  0
 14 31  1  0  0  0  0
 ```

The first block starting on line 5 contains the x, y, and z coordinates of every atom of the molecule in 3D space and the type of atom it is. The second block beginning on line 36 is the connection table. It tells us which atoms are bonded to each other and the type of bond it is. This will be important later on.

These two blocks are the information we wish to parse and store as our training data.

For now, we only need to focus on lines 1-65 to write our parsing script.

Looking through the file we see that the first line of every molecule block is just an increasing number. While we could use this number, we also notice that the second line is consistent between all molecule blocks in the file and doesn't change.

Let's begin writing our script and store this value as our starting point for each molecule block throughout the file.

```python
from periodictable import elements # We'll use this in a minute

filename = 'train.sdf' # The file we want the parse through
startstr = "  -OEChem"

with open(filename, 'r') as f:
    data = f.readlines()
```
So now we are opening the training file, reading all the lines and storing the data into a python list called data.

Line 4 of each molecule block in train.sdf also tells us important information. Specifically, it tells us the atom count and the number of connections.

Now we need to loop through our list and only look at the blocks of information we want for each molecule. We will loop through our start string and use that as our base index for each block


```python
for idx, line in enumerate(data) :
    if startstr in line :
        temp = data[idx + 2].split() # Get atom count and connections
        atmcount = temp[0]  # Store atom count
        n = len(atmcount)
        if n > 3 :  # if atom count is greater than 999
            numxyz = int(atmcount[0:n/2])
            numc = int(atmcount[n/2:n])
        else :
            numxyz = int(temp[0])
            numc = int(temp[1])
```
Okay, now we know the number of x, y, z coordinates we will have for the molecule and the number of connections.

Let's initialize some data structures to store this information.

```python
xyz = [[0.0 for i in range(3)] for j in range (numxyz)]
connect = [[0 for i in range(3)] for j in range(numc)]
atmnum = [0 for i in range(numxyz)]
```

Now that we have the atom count and the number of connections we'll use another call to split() to only look at the x, y, z, and atom type and store all this information into our data structures.

```python
for i in range(numxyz) :
    temp = data[idx + 3 + i].split()
    for el in elements :
        if temp[3] == el.symbol :
            atmnum[i] = el.number
    for j in range(3) :
        xyz[i][j] = float(temp[j])
```

As you can see, I also convert every atom symbol to it to the appropriate atomic number using the Periodic Table library that we imported at the beginning. This is optional, but since I am going to be putting all this into TensorFlow I would prefer to have everything as a number for simplicity later on.

Now, we store the connection table information.

```python
for i in range(numc) :
    temp = data[idx + 3 + i + numxyz].split()
    if len(temp) == 2 :
        atmcount = temp[0]
        n = len(atmcount)
        connect[i][0] = int(atmcount[0:n/2])
        connect[i][1] = int(atmcount[n/2:n])
        connect[i][2] = int(temp[1])
    else :
        connect[i][0] = int(temp[0])
        connect[i][1] = int(temp[1])
        connect[i][2] = int(temp[2])
```

Okay, everything should be parsed in now and stored as we want it. The last thing we need to do is print out all of this to the command line to make sure we did it right!

We will display everything in CSV format in case we want to store this later.

```python
for i in range(numxyz): print('{:d} {:s} {:12.5f} {:s} {:12.5f} {:s} {:12.5f}' .format( atmnum[i], ', ', xyz[i][0], ', ', xyz[i][1], ', ',xyz[i][2]) )

for i in range(numc): print('{:5d} {:s} {:5d} {:s} {:5d}' .format( connect[i][0], ', ', connect[i][1], ', ',connect[i][2]) )
```
Now let's save our file and run it like so:

```
python ./parse.py
```

And there we go! We can now parse through large SDF files and extract the information we want! Pretty cool eh? ;p

In my next post, I'll demonstrate how we are going to store this information in a way that we can use it as training data for our model in TensorFlow so stay tuned!
