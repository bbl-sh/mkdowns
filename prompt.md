# Bash Scripting Essentials for DevOps

Bash scripting is one of the most important skill in deevops. Learning it will definetely give you an upper hand.
[Here is my link to certification i did](https://drive.google.com/file/d/1iqvRKZ-0Se7YdxJ3Ix4CQqRl53qY5_P3/view?usp=sharing)
## Table of Contents
- [Variables and Inputs](#variables-and-inputs)
- [Looping (For Loop)](#looping-for-loop)
- [Case Statement](#case-statement)
- [Arrays](#arrays)
  - [Basic Arrays](#basic-arrays)
  - [Appending to Arrays](#appending-to-arrays)
  - [Reading Files into Arrays](#reading-files-into-arrays)
  - [Array Slicing](#array-slicing)
- [Functions](#functions)
  - [Function Arguments](#function-arguments)
  - [Default Arguments](#default-arguments)
  - [Input from Terminal](#input-from-terminal)
  - [Error Handling](#error-handling)
- [Enhancing Scripts with `tput`](#enhancing-scripts-with-tput)

---

## Variables and Inputs
```bash
#!/bin/bash
set -e  # Exit immediately on errors

name="aman"
echo "$name"

echo "First argument: $1"
echo "All arguments: $@"

read -p "Enter your name: " user_name
echo "Hello, $user_name"
```

---

## Looping (For Loop)
### a. Loop Over List
```bash
#!/bin/bash
set -e

for item in app cent pant; do
    echo "$item"
done
```

### b. C-style Loop
```bash
#!/bin/bash
set -e

for ((i=0; i<7; i++)); do
    echo "Siuuuuu"
done
```

### c. Loop Over Files
```bash
#!/bin/bash
set -e

mkdir -p test-files && touch test-files/{1..3}.txt

for file in test-files/*.txt; do
    echo "Found file: $file"
done
```

### d. Loop Command Output
```bash
#!/bin/bash
set -e

for id in $(seq 1 5); do
    echo "Processing ID: $id"
done
```

---

## Case Statement
```bash
#!/bin/bash
set -e

read -p "Enter number (1/7/10/17/18/45): " var

case "$var" in
    1)  echo "King is here" ;;
    7)  echo "Siuuuuu " ;;
    10) echo "Ankara messi ankara messi" ;;
    17) echo "ABD" ;;
    18) echo "Virat " ;;
    45) echo "RO superhit Sharma" ;;
    *)  echo "Please enter right input" ;;
esac
```

---

## Arrays
### Basic Arrays
```bash
#!/bin/bash
set -e

servers=("web01" "db01" "app01")
echo "All servers: ${servers[@]}"
```

### Appending to Arrays
```bash
servers+=("cache01")
echo "Updated list: ${servers[@]}"
```

### Reading Files into Arrays
```bash
mapfile -t lines < <(find . -type f)
printf "Found files:\n%s\n" "${lines[@]}"
```

### Array Slicing
```bash
echo "First 2 elements: ${servers[@]:0:2}"
```

---

## Functions
### Function Arguments
```bash
deploy() {
    local target="$1"
    echo "Deploying to: $target"
}
deploy "prod.example.com"
```

### Default Arguments
```bash
backup() {
    local dest="${1:-/backup}"
    echo "Backing up to: $dest"
}
backup
backup "/custom/path"
```

### Input from Terminal
```bash
prompt_deploy() {
    read -p "Enter server: " srv
    deploy "$srv"
}
prompt_deploy
```

### Error Handling
```bash
safe_command() {
    if ! command "$@"; then
        echo "Command failed: $*" >&2
        exit 1
    fi
}
safe_command ls /nonexistent
```

---

## Enhancing Scripts with `tput`
```bash
#!/bin/bash
set -e

tput bold && echo "IMPORTANT MESSAGE" && tput sgr0
tput cup 5 10
echo "Positioned text"
```
