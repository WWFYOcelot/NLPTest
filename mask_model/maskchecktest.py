def compare_with_mask(str1, str2):
    parts = str2.split("<mask>")
    pos = 0

    for part in parts:
        if part:
            pos = str1.find(part, pos)
            if pos == -1:
                return False
            pos += len(part)
    
    return True
# the latest: more homes razed by northern california wildfire - abc news http://t.co/ymy4rskq3d', 1) the latest: more homes razed by northern california wild<mask> - abc news http://t.co/ymy4rskq3d
# Example
str1 = "the latest: more homes razed by northern california wildfire - abc news http://t.co/ymy4rskq3d"
str2 = "the latest: more homes razed by northern california wild<mask> - abc news http://t.co/ymy4rskq3d"
print(compare_with_mask(str1, str2))  # True
