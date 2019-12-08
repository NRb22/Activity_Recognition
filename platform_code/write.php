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
    height: 760px;
    float: center;
    background-color: #505050;
  }
  </style>
  <body topmargion=0 leftmargion=0 text=#464646>
    <div id="container">
      <center>
        <font color='#19b092' >
      <br>
      <h1>
        LEARNINGEYES
      </h1>
      <img src="eye.png" width=20%/>
      <br>
      WORK JOURNAL
      <br><br><br><br>
      <form method="POST" action='insert.php' style="margin-left: -150px;margin-top: -30px;">
        <table>
          <tr>
            <td width=60 align=left>ID</td>
            <td align=left>
              <input type="text" name=name size=20>
            </td>
          </tr>
          <tr>
            <td width=60 align=left>password</td>
            <td align=left>
              <input type="password" name=pass size=20 maxlength=20>
            </td>
          </tr>
          <tr>
            <td width=60 align=left>Title</td>
            <td align=left>
              <input type="text" name=title size=67 maxlength=67>
            </td>
            <tr>
            <td width=60 align=left>content</td>
            <td align=left>
              <textarea name=content cols=100 rows=20></textarea>
            </td>
          </tr>
        </font>
          <td colspan=10 align=center>
            <input type="submit" value="save">
            &nbsp&nbsp
            <input type="reset" value="reset">
            &nbsp&nbsp
            <input type="button" value="back" onClick='history.back(-1);'>
          </td>
        </tr>
      </table>
    </form>
  </center>
</div>
</body>
</html>

