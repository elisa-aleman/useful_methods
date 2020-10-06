# Useful Methods
Useful methods I like to import to save time

## Special printing methods

* __up()__ : Go up a line in the terminal to print over something
* __down()__ : Go down a line in the terminal
* __print_STD_log(strlog, log_file)__ : Print string to terminal and to file at the same time
* __print_log(strlog, log_file)__ : Print string to a file
* __print_list_to_log(strlist, log_file)__ : Print list of strings to a file
* __print_list_to_STD_log(strlist, log_file)__ : Print list of strings to the terminal and to a file at the same time
* __write_tuple_list_to_csv(ins_table, filepath)__ : Write tuple list to CSV file
* __append_tuple_to_csv(ins_tuple, filepath)__ : Append tuple to CSV file

## Data organizing methods

* __tail(filepath, decode_utf8=True, with_head=True, pandas_read=True)__ : Returns the last line in a file
* __head(filepath, decode_utf8=True)__ : Returns the first line in a file
* __line_count_rawgen(filepath, last_index_csv=True)__ : Fast way to count lines in a file
* __read_dict(filename)__ : Get a word list text file (a word per line) into a list
* __flatten(container)__ : Make a list like [1,[2,3],[4,[5]]] into [1,2,3,4,5]

## CSV and SQL 

* __CSVcreateSQL(titles, dbname, tablename)__ : create empty SQL table from a csv file column titles
* __CSVtoSQL(titles, table, dbname, tablename)__ : copy csv into the created SQL table
* __SQLtoCSV(dbname)__ : make a csv by table from a SQL database
