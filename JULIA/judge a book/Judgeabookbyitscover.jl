using Flux
using Flux: @epochs, onehotbatch, crossentropy, throttle
using Images
using FileIO
using CSV
using Statistics

train_path = "G:\\mlimgs\\Book-Train-FULL\\"
train_csv = "F:\\book-dataset\\Task1\\book30-listing-train1.csv"

test_path = "G:\\mlimgs\\Book-Test-FULL\\"
test_csv = "F:\\book-dataset\\Task1\\book30-listing-test1.csv"

train_dataset  = CSV.read(train_csv)
test_dataset = CSV.read(test_csv)

train_imglist = readdir(train_path)
test_imglist = readdir(test_path)

train_setsize = length(train_imglist)
test_setsize = 500

batch_size = 400
imsize = 56
epochs = 10

function create_batch(indexs; path, csv, dataset)
    X = Array{Float32}(undef, imsize, imsize, 3, length(indexs))
    for (p,i) in enumerate(indexs)
        img = load(string(path,i,".png"))
        img = channelview(RGB.(imresize(img, imsize, imsize)))
        img = Float32.(permutedims(img, (2, 3, 1)))

        X[:, :, :, p] = img
    end
    Y = onehotbatch(dataset[indexs[1]:indexs[end], 1], 0:29)
    return (X, Y)
end

indexs = Base.Iterators.partition(1:train_setsize, batch_size)

test_set = create_batch(
    1:test_setsize;
    path = test_path,
    csv = test_csv,
    dataset = test_dataset
)
@info "creating the model"
m = Chain(
    Conv((3, 3), 3 => 32, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((3, 3), 32 => 64, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((2, 2), 64 => 128, pad = (1, 1), relu),
    MaxPool((2, 2)),

    Conv((2, 2), 128 => 256, pad = (1, 1), relu),
    MaxPool((2, 2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(4096, 1024, relu),
    Dropout(0.5),
    Dense(1024, 512),
    Dense(512, 30),
    softmax,
)
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(Flux.onecold(m(x)) .== Flux.onecold(y))

function cbfunc()
    ca = accuracy(test_set...)
    print("batch_acc: ",string(ca),"; ")
    cl = loss(test_set...)
    println("batch_loss: ",string(cl))
end

opt = ADAM(0.0001)

for e in 1:epochs
    @info "Epoch no.-> $e"
    b = 1
    for i in indexs
        println("Batch no. -> $b")
        train_batch = [create_batch(i; path = train_path, csv = train_csv, dataset = train_dataset)]
        Flux.train!(loss, params(m), train_batch , opt, cb = cbfunc)
        b+=1
    end

end

println("Final acc and loss : ")
cbfunc()
