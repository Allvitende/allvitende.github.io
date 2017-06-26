---
layout: post
title: Internship Update
---
This will be a relatively short post. As you may know, part of the motivation for starting this blog was to document my internship experience and to post some of the code that I have been writing. Since I don't have any new code to share yet (hopefully next week!) I'm just going to shed some light and random thoughts on what I have been up to so far.

After figuring out the best way to parse all of the data from for the training set, I have hit a bit of a wall. For this particular model that I am trying to build, the training set is not of a uniform size. In neural networks, most models have a defined size of input into the network this way everything is setup to correctly make the correlations. For example, you may have a 32 x 32 2D array for each image in a training data set if you were trying to detect symbols or handwriting. As such, you would build your model based on the known size of the input layer.

In my case it is a bit different. My inputs vary in size. The 2D array could be 25 x 25 or 50 x 50. I initially thought there would be a way to build a dynamic model based on this but it seems this is not trivial. A possible solution for this would be to use a recurrent neural network but I don't have any experience with implementing one of those yet and am on a pretty tight deadline to complete this project.

As a result, I am thinking of creating a maximum input size of say, 30 x 30 and anything below that size I would fill zeros in the matrix of euclidean distances as fillers and not take molecules greater than 30 x 30 into my training set. This should still produce the same desired outcome, it just is not as clean of a solution as I would have hoped for.

I'll be back next week with (hopefully!) some new code to share with you all. Thanks!
