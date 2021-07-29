import sqlite3 as sql

def insertUser(username,password):
    con = sql.connect("sensorsData.db")
    cur = con.cursor()
    cur.execute("INSERT INTO users (username,password) VALUES (?,?)", (username,password))
    con.commit()
    con.close()

def retrieveUsers():
	con = sql.connect("sensorsData.db")
	cur = con.cursor()
	cur.execute("SELECT username, password FROM users")
	users = cur.fetchall()
	con.close()
	return users

def registerUser(username):
	con = sql.connect("sensorsData.db")
	cur = con.cursor()
	cur.execute("SELECT * FROM users WHERE username = ?", (username,))
	users = cur.fetchall()
	con.close()
	return users

def final(username,password,email):
	con = sql.connect("sensorsData.db")
	cur = con.cursor()
	cur.execute("INSERT INTO users VALUES (NULL, ?, ?, ?)", (username, password, email,))
	con.commit()
	con.close()