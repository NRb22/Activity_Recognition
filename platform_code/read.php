<?php
  header('Content-Type: text/html; charset=utf-8');
?>
<?php session_start();
    DEFINE('DB_USERNAME', 'root');
    DEFINE('DB_PASSWORD', '');
    DEFINE('DB_HOST', 'localhost');
    DEFINE('DB_DATABASE', 'LearningEyes');
    $mysqli = new mysqli(DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASE);
    if (mysqli_connect_error()) {
        die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
    }
    ?>
    <!DOCTYPE html>
    <html>
    <head>
      <title>List</title>
      <style>
      body{
        background-color: #505050;
        <!--
        background-image: url('cctv.png');
        background-repeat:no-repeat;
        background-position: center;
        background-size:4000px 120%;
        -->
      }
      #container{
        margin: 0px auto;
        width: 530px;
        height: 780px;
        float: center;
        background-color: #505050;
      }
      #content{
        margin:0px auto;
        width: 600px;
        height: 400px;
        float: center;
        border:3px #19b092 solid;
	margin-left:-45px;
	margin-bottom: 5px;
      }
      </style>
      <body topmargion=0 leftmargion=0 text=#19b092>
        <div id="container">
          <center>
          <br>
          <h1>
            LEARNINGEYES
          </h1>
            <img src="eye.png" width=20%/>
            <br>
          WORK JOURNAL
          <?php
          $no=$_GET['id'];
          $sql="select * from board where id='$no'";
          $result=mysqli_query($mysqli,$sql);
          $row=mysqli_fetch_array($result);
          $view=$row['view']+1;
          $sql="update board set view='$view' where id='$no'";
          $result=mysqli_query($mysqli,$sql);
          $sql="select * from board where id='$no'";
          $result=mysqli_query($mysqli,$sql);
          $row=mysqli_fetch_array($result);
           ?>
           <h2>
           No:<?php echo $row['id'];?> TITLE: <?php echo $row['title'];?></h2>
           writer:<?php echo $row['userid'];?> date:<?php echo $row['wdate'];?> click:<?php echo $row['view'];?>
           <div id="content">
             <br>
           <?php echo $row['content'];?>
         </div>
         <form method="POST" action="delete.php">
         <input type="text" name="no" size="2"value="No">
         <input type="text" name="pass" value="enter password">
         <input type="submit" value="delete">
       </form>
      </center>
    </div>
    </body>
    </html>
