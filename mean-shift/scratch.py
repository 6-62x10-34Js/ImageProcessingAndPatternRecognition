Input: image, bandwidth

Initialize
empty
list
modes

For
each
pixel(x, y) in the
image:
Initialize
mode
m
to
the
RGB
color
of
the
pixel
Initialize
empty
list
neighbors

For
each
pixel(a, b) in the
image:
If
the
Euclidean
distance
between(x, y) and (a, b) is less
than
the
bandwidth:
Add(a, b)
to
the
list
of
neighbors

While
the
mode
m
has
changed:
Initialize
empty
list
new_neighbors
For
each
pixel(a, b) in the
list
of
neighbors:
If
the
Euclidean
distance
between
m and the
RGB
color
of(a, b) is less
than
the
bandwidth:
Add(a, b)
to
the
list
of
new_neighbors
Update
the
mode
m
to
the
mean
RGB
color
of
the
pixels in the
list
of
new_neighbors
Update
the
list
of
neighbors
to
the
list
of
new_neighbors

If
the
mode
m is not already in the
list
of
modes:
Add
m
to
the
list
of
modes

For
each
pixel(x, y) in the
image:
Set
the
RGB
color
of(x, y)
to
the
nearest
mode in the
list
of
modes

Output: segmented
image
