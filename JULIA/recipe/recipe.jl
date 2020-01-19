using Plots
inspectdr() # For an unknown reason really speeds up the plotting process

# We're defining a Type Recipe as it is one of the easiest to create
@recipe function f(equation::String, xr = 1:10)
    linecolor   --> :blue # The special operator --> turns linecolor --> :blue into get!(plotattributes, :linecolor, :blue)
                          # setting the attribute only when it doesn't already exist
    seriestype  :=  :path # The special operator := turns seriestype := :path into plotattributes[:seriestype] = :path
                          # forcing that attribute value.
    # The lines below treat the string equation as an actual equation by subsituting a value in it
    # and getting the output

    equ = split(equation, "")
    y = Vector()
    for num in xr
        eq = copy(equ) # we create a fresh copy of the equation every new number
                       # this is because we subsitute x with the current num every iteration of the for loop.
                       # and we'll require a fresh copy every iteration
        for i in 1:length(equ)
            eq[i] == "x" ? eq[i] = string(num) : nothing # if we see 'x' replace it with num
        end
        eq = join(eq)
        push!(y,eval(Meta.parse(eq))) # we are pushing the computed y value into a vector to be plotted later on
    end

    xr, y # returning the stuff to be plotted
end

plot("x^3 + 10^4", -100:100)





# It took me a while to grasp the concept of the different types of recipes.
# The docs we not able to explain very well what is the distinction between each type.
# Examples and more case studies would be great. However, in some examples I was unsure of the
# stuff happening, and it wasn't explained. For example, we sometimes return () from series recipe
# which is very different from the return of other types of recipes. After a lot of tinkering around
# and playing with Plots I was able to figure out why that was. 
