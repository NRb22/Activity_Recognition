<?php
    $db = new mysqli('localhost', 'root', '', 'LearningEyes');
    $db->query('SET NAMES utf8');
?>
<!DOCTYPE html>
<style>

body{
  background-color: #505050;
}
#container{
  margin: 0px auto;
  width: 350px;
  height: 220px;
  float: center;
  background-color: #505050;
}
</style>
<head>
  <meta charset="utf-8"/>
  <title> SIGN IN </title>
</head>
<body>
  <div id="container">
    <center>
      <br><br><br>
      <font color=#19b092>
    <h1> LEARNING EYES</h1>
    <img src="eye.png" width=100%/>
  </center>
    <form method="post" action="signin_ok.php">
    </br>
      <fieldset style="border:3 solid; border-top-color:#19b092; border-bottom-color:#19b092; border-left-color:#19b092; border-right-color:#19b092; width: 360px;">
        <legend> 입력사항</legend>
        <table style="margin-left: 15px;">
          <tr>
            <td> ID</td>
            <td><input type="text" size="30" name="userid" placeholder="아이디"></td>
          </tr>
          <tr>
            <td> PW</td>
            <td><input type="password" size="30" name="userpw" placeholder="비밀번호"></td>
          </tr>
          <tr>
            <td>manager ID&nbsp</td>
            <td><input type="text" size="30" name="name" placeholder="이름"></td>
          </tr>
        </table>
        <div style="margin-left: 140px; margin-top: 5px;">
          <input type="submit" value="signin"/> 
          <input type="reset" value="reset"/>
        </div>
      </fieldset>
    </form>
  </body>
  </html>

