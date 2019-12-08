<?php session_start() ;
  
  $db = new mysqli('localhost', 'root', '', 'LearningEyes');
  $db->query('SET NAMES utf8');
	error_reporting(0);
?>
<!DOCTYPE html>
<html>
   <meta charset="utf-8">
   <head>
     <link href="https://fonts.googleapis.com/css?family=Roboto&display=swap" rel="stylesheet">
     <style>
     body{
       background-color:#505050;
       <!--
       background-image: url('cctv.png');
       background-repeat:no-repeat;
       background-position: center;
       background-size:1600px 600%;
       -->

     }
     #container{
       margin: 0px auto;
       width: 250px;
       height: 500px;
       float: center;
       background-color: #505050;
       font-family:'Roboto';
     }
     </style>
   </head>
   <body>
     <font color='#19b092' >
      <center>
        <div id="container">
            <br><br><br><br>
	<h1>LEARNING EYES</h1><br/>
  <img src="eye.png" width=100%/>
	<p>
	This is login page<br/>
	Please enter your id and password<br/>
	</p>
          <form method="POST" action="login.php">
            <p style="display: inline-block; text-align: left;">ID : </p>
         	  <input type="text" maxLength="20" name="userid">
            <div></div>
         	  <p style="display: inline-block; text-align: left; margin-top: -20px;" >PW : </p>
            <input type="password" maxLength="20" name="password" style="margin-right: 10px;"><br/>
         	<input type="submit" value="login">
        </form>
          <tr>
          </br>
            <td clospab="3" aligh="center" class="mem">
              <a href="signin.php"> sign in?</a>
            </td>
          </tr>
      </center>
    </div>
  </font>
   </body>
</html>

