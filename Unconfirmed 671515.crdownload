import tkinter as tk

# Function to update expression in the display
def press(num):
    global expression
    expression = expression + str(num)
    equation.set(expression)

# Function to evaluate the final expression
def equalpress():
    try:
        global expression
        total = str(eval(expression))   # evaluate string expression
        equation.set(total)
        expression = total              # allow further calculation
    except:
        equation.set(" error ")
        expression = ""

# Function to clear the screen
def clear():
    global expression
    expression = ""
    equation.set("")

# Main program
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Simple Calculator")
    root.geometry("550x400")   # Bigger window size
    root.resizable(False, False)

    expression = ""
    equation = tk.StringVar()

    # Display area (Bigger Screen)
    display = tk.Entry(root, textvariable=equation, font=("Arial", 24),
                       bd=10, insertwidth=2, width=25,
                       borderwidth=4, relief="ridge", justify="center")
    display.grid(row=0, column=0, columnspan=4, ipadx=8, ipady=20)

    # Buttons layout
    buttons = [
        ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
        ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
        ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
        ('C', 4, 0), ('0', 4, 1), ('=', 4, 2), ('+', 4, 3),
    ]

    for (text, row, col) in buttons:
        if text == "=":
            btn = tk.Button(root, text=text, fg="black", bg="lightgray",
                            font=("Arial", 18), height=2, width=8,
                            command=equalpress)
        elif text == "C":
            btn = tk.Button(root, text=text, fg="black", bg="lightgray",
                            font=("Arial", 18), height=2, width=8,
                            command=clear)
        else:
            btn = tk.Button(root, text=text, fg="black", bg="lightgray",
                            font=("Arial", 18), height=2, width=8,
                            command=lambda t=text: press(t))
        btn.grid(row=row, column=col, padx=5, pady=5)

    root.mainloop()
