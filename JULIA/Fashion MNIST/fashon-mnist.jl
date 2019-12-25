# This file assumes that the readers are
# aware of the very core concepts of neural networks and ML which are not specific to julia

using Flux # Can't do machine learning in flux whithout flux :)
using Statistics # To calculate the mean which is required for finding the accu
using Flux: @epochs, onehotbatch, onecold, crossentropy, throttle # The uses of these will be explained later
using Base.Iterators: repeated # This will be used to make mulitple copies of the dataset

using MLDatasets # This library allows gives us access to
                 # various datasets and in the most convenient format possible.

using BSON: @save # This is to save the model


imgs, labels = FashionMNIST.traindata() # Load the dataset, current shape of imgs is (28, 28, 60000).
                                        # the shape of labels is (60000,).
                                        # It is just a big vector of numbers corresponding to a image in the images array
test_images, test_labels = FashionMNIST.testdata()


# Now the main intent of using MLDatasets comes to light.
# we dont have to make another array of the correct dimentions and populate \
# it will data. Now we can just directly use hcat after reshaping it to do the job
X=hcat(float.(reshape(imgs, 28^2, 60000)))
# Inputing a multidimentional array to the neurons
# of the model is computationally very ineffitient and generally
# has been found to not produce great results. This is why we are
# "Flattening" the images. This just putting all the
# data in the image into a 1D vector


# onehotencoding turns (usually) neumerical data into kind of truth tabels.
# example, onehot(:b, [:a, :b, :c])
# 3-element Flux.OneHotVector:
# false
#  true
# false
# onehotbatch just one hot encodes all the data in a provided array
# and yields the encoding of every element in an output OneHotMatrix.
Y = onehotbatch(labels, 0:9)


# constuction of the model
m = Chain(
  Dense(28^2, 128, relu),
  Dense(128,32, relu),
  Dense(32, 10),
  softmax)


loss(x, y) = crossentropy(m(x), y) # Finds the crossentropy between the computed value of x and the actual value of x
                                   # what crossentropy is and how it is computed is beyond the scope of this file. Sorry :(

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y)) # onecold does what is sounds like it might do
# it turns one hotencoded vectors into the orignal data
# People might wonder why are we using onecold over here?
# there are two reasons that I can think of for using onecold.
# One is that m(x) has its last layer as softmax. This means it returns a probability distribution.
# As it turns out, onecold works with probablity distributions too.
# this means it turns the PB into normal neumerical data which can be used with the == sign.
dataset = [(X,Y)]
losses = []
accu = []
function evalcb()
    # accu = global accu;losses=global losses
    current_acc = accuracy(X,Y)
    @show(current_acc)
    current_loss = loss(X, Y)
    @show(current_loss)
    push!(accu,current_acc)
    push!(losses,current_loss)

end

opt = ADAM(0.01 )
@epochs 30 Flux.train!(loss, params(m), dataset, opt, cb = evalcb) # throttle is a function which takes another function
                                                                        # which it calls every n seconds. n is 10 in this case.

accuracy(X, Y)

# Test set accu
tX = hcat(float.(reshape(test_images, 28^2, 10000)))
tY = onehotbatch(test_labels, 0:9)

history = (losses, accu)
println("Final acc-> ",accuracy(tX, tY))
@save "model.bson" m
@save "juliahistory.bson" history
