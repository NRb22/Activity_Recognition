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
    $id=$_GET[id];
    $userid=$_POST[name];
    $pass=$_POST[pass];
    $title=$_POST[title];
    $content=$_POST[content];
    $sql="insert into board(userid,pass,title,content,wdate) values ('$userid','$pass','$title','$content',now())";
    $result=mysqli_query($mysqli,$sql);
    if($result==false){
      echo mysqli_error($mysqli);
    }
    echo"<meta http-equiv='Refresh' content='0; URL=main.php'>";
    mysqli_close($mysqli);

    ?>
