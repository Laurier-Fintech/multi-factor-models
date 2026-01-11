def cusip_check_digit(cusip8):
    values = {ch: i for i, ch in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ*@#', start=0)}
    total = 0
    for i, ch in enumerate(cusip8):
        v = values[ch]
        if (i + 1) % 2 == 0:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - (total % 10)) % 10)

# Convert from unique CUSIP file "unique_cusips.txt" to full CUSIP file "full_cusip.txt"
# with open("./outputs/unique_cusip_codes.txt", "r") as infile, open("./outputs/full_cusip_codes.txt", "w") as outfile:
#     for line in infile:
#         cusip8 = line.strip()
#         print(cusip8 + '\n')
#         full_cusip = cusip8 + cusip_check_digit(cusip8)
#         outfile.write(full_cusip + "\n")              

cusip_convert = "36720410"

print("Full CUSIP:", cusip_convert + cusip_check_digit(cusip_convert))