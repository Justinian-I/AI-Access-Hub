#importing re module
import re
#creating a function for name taking name as an argument
def blankname(name):
   #assign blank variable as false
   blank = False
   #checking if name entered by user is blank
   if name == "":
      #if blank give a message
      blank = {"message":"Please enter name"}
   else:
      #if not blank it is true
      blank = True
   return blank

def user(name):
   blank = False
   if name == "" :
      blank = {"message":"No blank or duplicate entry"}
   else:
      blank = True
   return blank

#from re module taking regex operations
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
#creating a function for email taking email as an argument
def checkemail(email):
   #assign valid variable as false
   valid  =  False
   #if email has all essentials
   if (re.fullmatch(regex, email)):
      valid = True
   else:
      #if not give a message
      valid = {"message":"invalid email"}

   return valid

#creating a function for password taking password as an argument
def passwordcheck(password):
   #assign valid variable as false
   valid = False
   #taking max lenght as 8
   lenght = 8
   #if password lenght is blank or less than 8 characters
   if password == "" or  len(password) < lenght:
      #give a message
      valid = {"message":"password minimum 8 characters"}
   else:
      valid = True

   return valid

#creating a function for mobile no taking mobile_no as an argument
def mobileno(mobile_no):
   valid = False
   # mobile no has all essentials and is correct, aasign to r variable
   r = re.fullmatch('[6-9][0-9]{9}' , mobile_no)
   #if r is not equal to none
   if r!= None:
      valid = True
   else:
      #if not give a message
      valid = {"message":"Not a valid number"}

   return valid

#creating a function for mobile no taking mobile_no as an argument
def checkbio(bio):
   valid =  False
   #if len of bio is less than 30 characters or blank
   if len(bio) > 30 or bio == "":
      valid = {"message":"Blank bio or keep it shorter"}
   else:
        valid = True

   return valid

#creating a function validate and taking info(we passed in signup route) as an argument
def validate(info):
   condition = False
   #assign function(blankname(name)) to a variable(validateName)
   #here blankname will take database fname from info 
   validateName = blankname(info["fname"])                                                                     
   validateUsername = user(info["fusername"])
   validateEmail = checkemail(info["femail"])
   validatePassword = passwordcheck(info["fpassword"])
  


   #check if validateName is true
   if validateName == True:
      #check if validateUsername is true                                                                           
      if validateUsername == True:                                                                     
         if validateEmail == True:
            if validatePassword == True:
               
                  
                     #if all are true good to go
                     condition = True                                                                  
                                                                                                      
                                                                             
                                                                         
            else:
               ##condition is false & will print message of validatePassword function
               condition = validatePassword
         else:
            #condition is false & will print message of validateEmail function
            condition = validateEmail
      else:
         #condition is false & will print message of validateUsername function
         condition = validateUsername
   else:
      #condition is false & will print message of validateName function
      condition = validateName
   return condition


def validateLogin(desc):
   condition = False
   #assign function checkemail to validateEmail
   #here checkemail will take database femail from desc
   validateEmail = checkemail(desc["femail"])
   #assign function passwordcheck to validatePass                               
   validatePass = passwordcheck(desc["fpassword"])
   #check if validateEmail is true                          
   if validateEmail == True:                                               
      if validatePass == True:
         condition = True
      else:
         condition = validatePass
   else:
      condition = validateEmail
   return condition