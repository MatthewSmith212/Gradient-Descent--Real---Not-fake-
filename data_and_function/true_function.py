import math

def baseFunction(hours_spent : float) -> float:
    #student dies
    if hours_spent > 40:
        return 0.0 # kinda pollutes the data
        pass

    terms = list()
    #terms are coefficient, exponenet (variable is hours_spent)
    terms.append((60, 0))
    terms.append((17.481350, 1))
    terms.append((-3.378873, 2))
    terms.append((0.226777, 3))
    terms.append((-0.004827, 4))

    result = float()
    for term in terms:
        coefficient, exponent = term[0], term[1]
        result += coefficient * math.pow(hours_spent, exponent)
    return result

if __name__ == "__main__":
    results = list()
    