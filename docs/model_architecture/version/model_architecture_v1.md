# Sign Language Recognizer - Model architecture

The sign language recognizer model is a Classifier model.

It takes a bunch of coordinates that represent an element in 3D.
(e.g: a hand, a body, etc...)
And

> Keep in mind that this model architecture is unable to tell the framerate.

## Input
The model as for input:

**(21_DataPoint * 3_3dCoordinate)**

The input should look like something like that:
```c
// => : Neuron
// === : Other/Hidden layers
// ... : Other input neuron

For each data point:
    x => ===
    y => ===
    z => ===
...
```

## Output
The model as for output X neuron. Where X represent the number of sign the model can recognize (Not recognizing a sign is considered as case where the model recognize there's no sign).
The output should look like something like this:
```c
// If the model is capable of recognizing for example: ["No sign", "a", "b", "c"]
// There will be 4 output neuron the following way:
// => : Neuron
// === : Other/Hidden layers

=== => "No sign"
=== => "a"
=== => "b"
=== => "c"
```
