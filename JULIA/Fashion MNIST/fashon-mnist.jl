# This file assumes that the readers are
# aware of the very core concepts of neural networks and ML which are not specific to julia

using Flux # Can't do machine learning in flux whithout flux :)
using Statistics # To calculate the mean which is required for finding the accu
using Flux: @epochs, onehotbatch, crossentropy, throttle, Data.FashionMNIST # The uses of these will be explained later
using Base.Iterators: partition # This will be used to make mulitple copies of the dataset

using BSON: @save # This is to save the model


imgs = FashionMNIST.images()  # Load the data
labels = FashionMNIST.labels()

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs)) # The orignal dataset , X, is not of the right shape.
                                                                    # so it cant be just divided into smaller parts.
                                                                    # For this, we are creating another array of the correct shape which will be our batch.
                                                                    # and we will fill this array with the data from our dataset.
                                                                    # The correct shape is (IMG_DATA, IMG_DATA, COLOR_CHANNEL, SAMPLES)

                                                                    # A good way to think about this is that every image is of the shape 28, 28
                                                                    # which is 28 rows and 28 column of pixles. Flux requires us to add another dim which represents the color.
                                                                    # So an BLACK AND WHITE image would be like a cuboid of thickness 1, hight and width of 28.
                                                                    # An RGB image would be like 3 cuboids each representing red, blue and green channels respectivly. Each having
                                                                    # thickness 1 and hight and width of 28. This type of thinking is good for beginners as it helps visualize things.
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)# onehotencoding turns (usually) neumerical data into kind of truth tabels.
    # example, onehot(:b, [:a, :b, :c])
    # 3-element Flux.OneHotVector:
    # false
    #  true
    # false
    # onehotbatch just one hot encodes all the data in a provided array
    # and yields the encoding of every element in an output OneHotMatrix.


    return (X_batch, Y_batch)
end

batch_size = 500
mb_idxs = partition(1:length(imgs), batch_size) # We are dividing an array of numbers of the size
                                                # of the imgs into the size of batches. This array of arrays of
                                                # numbers will be used as indexs.

train_set = [make_minibatch(imgs, labels, i) for i in mb_idxs] # Every array in mb_idxs is an array of the indexs
                                                               # of the images which we want to be in that batch.
                                                               # so we are calling the func minibatch on that array.

test_imgs = FashionMNIST.images(:test) # By doing this we are getting all the images which are in the testing set
test_labels = FashionMNIST.labels(:test)

test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs)) # Now here we are specifing the batch to be the whole testing set

train_set = gpu.(train_set) # load everything in the arrays on gpu
test_set = gpu.(test_set)

# constuction of the model
m = Chain(
# First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>64, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 64, N)
    # which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(576, 10),

    # Finally, softmax to get nice probabilities
    softmax,)
m = gpu(m) # load the model on gpu

m(train_set[1][1]) # pre-compile the model to make calculations a bit faster

loss(x, y) = crossentropy(m(x), y) # Finds the crossentropy between the computed value of x and the actual value of x
                                   # what crossentropy is and how it is computed is beyond the scope of this file. Sorry :(

accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y)) # onecold does what is sounds like it might do
# it turns one hotencoded vectors into the orignal data
# People might wonder why are we using onecold over here?
# there are two reasons that I can think of for using onecold.
# One is that m(x) has its last layer as softmax. This means it returns a probability distribution.
# As it turns out, onecold works with probablity distributions too.
# this means it turns the PB into normal neumerical data which can be used with the == sign.
losses = []
accu = []
function evalcb()
    # accu = global accu;losses=global losses
    current_acc = acc = accuracy(test_set...)
    @show(current_acc)
    current_loss = loss(test_set...)
    @show(current_loss)
    push!(accu,current_acc)
    push!(losses,current_loss)

end

opt = ADAM()
@epochs 5 Flux.train!(loss, params(m), train_set, opt, cb = evalcb)

history = (losses, accu) # we are saving the losses and acuracys for plotting perposes.
println("Final acc-> ",accuracy(test_set...))
@save "juliahistory.bson" history
