
println("ROCK PAPER TIME!!!")
println("Youl'll be playing rock paper and scissors with the cpu!\n enter rock paper or scissor after shoot to input your choice!")
choices = ["Rock", "Paper","Scissors"]
turns = 1
player = 0
cpu = 0

function intro()
    println("\n ---- \n")
    for i in choices
        println(i)
        sleep(0.5)
    end
    println("Enter!")
    println("\n ---- \n")
end




function lose(cpu_choice)
    global cpu+=1
    println("CPU chose - > "*cpu_choice)
    println("YOU LOSE")
end

function win(cpu_choice)
    global player+=1
    println("CPU chose - > "*cpu_choice)
    println("YOU WIN!")
end

function  rerole()
    println("\n ---- \n")
    println("YOU AND THE CPU GOT THE SAME THING\nREROLE!")
    println("\n ---- \n")
    intro()
    turn = lowercase(readline())
    cpu_choice = lowercase(rand(choices))
    cpu_choice == turn ? cpu_choice=rerole() : nothing
    return cpu_choice
end

for _ in 1:3
    player = 0
    cpu = 0
    intro()
    turn = lowercase(readline())
    cpu_choice = lowercase(rand(choices))
    cpu_choice == turn ? cpu_choice=rerole() : nothing
    cpu_choice == "rock" && turn == "scissors" ? lose(cpu_choice) : nothing
    cpu_choice == "paper" && turn == "rock" ? lose(cpu_choice) : nothing
    cpu_choice == "scissors" && turn == "paper" ? lose(cpu_choice) : nothing

    turn == "rock" && cpu_choice == "scissors" ? win(cpu_choice) : nothing
    turn == "paper" && cpu_choice == "rock" ? win(cpu_choice) : nothing
    turn == "scissors" && cpu_choice == "paper" ? win(cpu_choice) : nothing

end
println("\n\nPLAYER SCORE -> "*string(player))
println("CPU SCORE -> "*string(cpu,"\n"))

player > cpu ? println("YOU WIN THE GAME!") : println("\n\nYOU LOSE THE GAME!")

