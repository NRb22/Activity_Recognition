<?php
  header('Content-Type: text/html; charset=utf-8');
?>
<?php session_start();
$mysqli = new mysqli('localhost', 'root', '', 'LearningEyes');
$mysqli->query('SET NAMES utf8');

    if (mysqli_connect_error()) {
        die('Connect Error ('.mysqli_connect_errno().') '.mysqli_connect_error());
    }
   	//Index.php에서 받은 userid, password가 설정되어있는지 확인 (if)

    $userid=$_POST['userid'];
    $userpw=password_hash($_POST['userpw'],PASSWORD_DEFAULT);
    $manager=$_POST['name'];
    $sql="select empno,userid from employee where userid='$manager'";
    $result=mysqli_query($mysqli,$sql);
    $row=mysqli_fetch_array($result);

    if($row['userid']==$manager){
      $empno=$row['empno'];
      $qry="insert into employee (userid, password,manager) values ('$userid','$userpw','$empno')";
      $result=mysqli_query($mysqli,$qry);
      if($result==false){
        echo mysqli_error($mysqli);
      }
      echo "<script>location.href='index.php';</script>";
    }
    ?>
