## Regular Expression
### Quick Guide
Symbol | Quick Guide | Example
--- | --- | ---
^ | Matches the `beginning` of a line | `^X.*:` => `X-Sieve:` CMU Sieve 2.3
$ | Matches the `end` of the line | ---
. | Matches `any` character | ---
\s | Matches `whitespace` | ---
\S | Matches any `non-whitespace` character | `^X-\S+:` !=>`X-Plane is behind schedule:` two weeks
* | `Repeats` a character zero or more times | ---
*? | `Repeats` a character zero or more times (non-greedy) | ---
+ | `Repeats` a character one or more times | ---
+? | `Repeats` a character one or more times (non-greedy) | ---
[aeiou] | Matches a single character in the listed `set` | ---
[^XYZ] | Matches a single character `not in` the listed `set` | ---
[a-z0-9] | The set of characters can include a `range` | ---
( | Indicates where string `extraction` is to start | ---
) | Indicates where string `extraction` is to end | ---

- re.search()
    - returns True or False
    ```py
    if re.search('^Frome:', line):
        print (line)

- re.findall()
    - want the matching strings to be extracted
    ```py
    x = 'My 2 favorite numbers are 19 and 42'
    y = re.findall('[0-9]+', x)
    print (y)
    # result is ['2', '19', '42']

### Greedy/ Non-Greedy Matching
- Greedy: * and + push `outward` in both directions to match the largest possible string
    ```py
    x = 'From: Using the : character'
    y = re.findall('^F.+:', x)
    print (y)
    # output is: ['From: Using the :']

- Non-Greedy: add a ? character, the + and * chill out a bit
    ```py
    x = 'From: Using the :character'
    y = re.findall('^F.+?:', x)
    print (y)
    # output is: ['From:']

### String Extraction
- Parentheses are not part of the match, but they tell where to `start` and `stop` what string to extract
    - ex1
        ```py
        x = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
        y = re.findall('\S+@\S+', x)
        print (y)
        # output is: ["stephen.marquard@uct.ac.za"]

        y = re.findall('^From (\S+@\S+)', x)
        print (y)
        # output is ["stephen.marquard@uct.ac.za"]

    - ex2 
        ```py
        x = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
        y = re.findall('@([^ ]*)', x) # [^ ]match non-blank character
        print (y)
        # output is ['uct.ac.za']

### Escape Character
- want a special regular expression character to just behave `normally`, you prefix it with '\'
    ```py
    x = 'We just received $10.00 for cookies.'
    y = re.findall('\$[0-9.]+', x)
    print (y)
    # output is ['$10.00']
