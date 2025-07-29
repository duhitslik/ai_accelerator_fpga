# txt_to_mem.py
def txt_to_mem(txt_path, mem_path, bit_width=8):
    with open(txt_path, 'r') as f_in, open(mem_path, 'w') as f_out:
        for line in f_in:
            tokens = line.strip().split()
            for token in tokens:
                if token == '': continue
                val = int(token)
                if val < 0:
                    val = (1 << bit_width) + val  # Convert to 2's complement
                hex_str = format(val, '0{}X'.format((bit_width + 3) // 4))  # Hex format
                f_out.write(hex_str + '\n')

# Convert all files
txt_to_mem("input.txt", "input.mem")
txt_to_mem("w1.txt", "w1.mem")
txt_to_mem("b1.txt", "b1.mem")
txt_to_mem("w2.txt", "w2.mem")
txt_to_mem("b2.txt", "b2.mem")
